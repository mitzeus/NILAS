import os
import math
import asyncio
import multiprocessing
from itertools import product

from vllm import SamplingParams
from src.corrector.multiagentic_judge.run import EvaluationConfig
from src.helper.typing import WordConstraintType

import traceback


async def process_work_items(
    gpu_id: int,
    work_items: list,
    generator_map: dict,
    generator_system_prompt,
    config: EvaluationConfig,
    custom_sampling_params: SamplingParams,
    max_response_token_length: int,
    use_word_constraint: bool,
    word_constraint_type: WordConstraintType,
    prompt_allowed_words: bool,
):
    from src.corrector.multiagentic_judge.run import run_self_iteration

    for lang, prompt, prompt_i, sample_id, batch_size in work_items:
        # Each iteration:
        # 1. blocks on vLLM generate()
        # 2. fans out async API calls in parallel
        # 3. awaits synthesis
        # 4. moves to next item
        try:
            await run_self_iteration(
                generator_system_prompt,
                prompt,
                generator_map[lang][batch_size],
                config,
                custom_sampling_params,
                max_response_token_length,
                lang,
                prompt_i,
                sample_id,
                batch_size,
                use_word_constraint,
                word_constraint_type,
                prompt_allowed_words=prompt_allowed_words,
                verbose="sequence",
                gpu_id=gpu_id,
            )
        except Exception:
            traceback.print_exc()


def gpu_worker(
    gpu_id: int,
    work_items: list,
    # vLLM / model config
    model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enable_prefix_caching: bool,
    logprobs: int,
    # Custom vLLM constructor
    lemmatizers: dict,
    allowed_words: dict[str, dict[int, list[str]]],
    batch_sizes: list[int],
    beam_size: int,
    use_word_constraint: bool,
    word_constraint_type: WordConstraintType,
    word_soft_constraint_penalty: float,
    prompt_allowed_words: bool,
    alpha: float,
    # run config
    generator_system_prompt: str,
    config: EvaluationConfig,
    custom_sampling_params: SamplingParams,
    max_response_token_length: int,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from src.ai.vLLM import initialize_vLLM
    from src.ai.models import Custom_vLLM
    from src.corrector.multiagentic_judge.run import run_self_iteration
    from vllm.transformers_utils.tokenizer import get_tokenizer

    # Reinitialize tokenizer in worker process (prevents multiprocessing serialization issues)
    tokenizer = get_tokenizer(model_name)

    # Initialize vLLM for worker instance
    llm = initialize_vLLM(
        model=model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        max_logprobs=logprobs + 20,
        trust_remote_code=True,
    )

    # Build one Custom_vLLM per language
    generator_map = {
        lang: {
            batch_size: Custom_vLLM(
                model=llm,
                tokenizer=tokenizer,
                lemmatizer=lemmatizers[lang],
                language=lang,
                allowed_words=allowed_words[lang][batch_size],
                beam_size=beam_size,
                word_soft_constraint_penalty=word_soft_constraint_penalty,
                alpha=alpha,
            )
            for batch_size in batch_sizes
        }
        for lang in lemmatizers
    }

    print(f"[GPU {gpu_id}] Online ー {len(work_items)} items to process")

    asyncio.run(
        process_work_items(
            gpu_id,
            work_items,
            generator_map,
            generator_system_prompt,
            config,
            custom_sampling_params,
            max_response_token_length,
            use_word_constraint,
            word_constraint_type,
            prompt_allowed_words,
        )
    )

    print(f"[GPU {gpu_id} Done.]")


def run_parallel(
    prompts_by_language: dict,
    batch_sizes: list[int],
    model_sample_size: int,
    num_gpus: int,
    # vLLM config
    model_name: str,
    dtype: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    enable_prefix_caching: bool,
    logprobs: int,
    # Custom vLLM Constructor
    tokenizer,
    lemmatizers: dict,
    allowed_words: dict[str, dict[int, list[str]]],
    beam_size: int,
    use_word_constraint: bool,
    word_constraint_type: WordConstraintType,
    word_soft_constraint_penalty: float,
    prompt_allowed_words: bool,
    alpha: float,
    # run config
    generator_system_prompt: str,
    config: EvaluationConfig,
    custom_sampling_params: SamplingParams,
    max_response_token_length: int,
):
    # get all units of work
    all_work: list[tuple] = []

    for lang, prompts in prompts_by_language.items():
        for prompt_i, prompt in enumerate(prompts):
            for batch_size in batch_sizes:
                for sample_id in range(model_sample_size):
                    all_work.append((lang, prompt, prompt_i, sample_id, batch_size))

    total = len(all_work)
    print(f"Total work items: {total} | GPUs {num_gpus}")

    chunk_size = math.ceil(total / num_gpus)
    work_slices = [
        all_work[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus)
    ]

    ctx = multiprocessing.get_context("spawn")
    processes = []

    for gpu_id, slice_ in enumerate(work_slices):
        if not slice_:
            continue

        p = ctx.Process(
            target=gpu_worker,
            kwargs=dict(
                gpu_id=gpu_id,
                work_items=slice_,
                model_name=model_name,
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                logprobs=logprobs,
                generator_system_prompt=generator_system_prompt,
                lemmatizers=lemmatizers,
                allowed_words=allowed_words,
                batch_sizes=batch_sizes,
                beam_size=beam_size,
                use_word_constraint=use_word_constraint,
                word_constraint_type=word_constraint_type,
                word_soft_constraint_penalty=word_soft_constraint_penalty,
                prompt_allowed_words=prompt_allowed_words,
                alpha=alpha,
                config=config,
                custom_sampling_params=custom_sampling_params,
                max_response_token_length=max_response_token_length,
            ),
        )
        p.start()
        processes.append((gpu_id, p))
        print(f"[GPU {gpu_id}] Spawned ー {len(slice_)} items")

    failed = []
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append(gpu_id)
            print(f"[GPU {gpu_id}] ✗ Worker exited with code {p.exitcode}")
        else:
            print(f"[GPU {gpu_id}] ✓ Finished cleanly")

    if failed:
        raise RuntimeError(f"Workers on GPU(s) {failed} failed. Check logs above.")

    print("All workers done.")
