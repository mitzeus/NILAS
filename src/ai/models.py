from vllm import LLM, SamplingParams
from src.ai.beam_search import BeamSearch
from vllm.inputs import TokensPrompt
import contractions
import re


def generate_token(
    prompt_token_ids: list[int] | list[list[int]],
    model: LLM,
    sampling_params: SamplingParams,
    verbose_generation: bool = False,
) -> tuple[dict[int, any], int, int]:
    """
    Generates a single token given the prompt and chat history with vLLM.

    # TODO change documentation to reflect also can take multiple prompts and give multiple returns
    Args:
        prompt_token_ids: A list of token IDs representing the input prompt.
        model: The vLLM model to use for generation.
        sampling_params: Parameters for sampling strategy.

    Returns:
        dict: Logprobs
        int: Sampled token ID
        int: Greedily selected token ID
    """

    do_multiprocess = True if isinstance(prompt_token_ids[0], list) else False

    if do_multiprocess:
        # Process and return several token generations in parallel (for beam search)

        batched_prompts = [
            TokensPrompt(prompt_token_ids=single_prompt_ids)
            for single_prompt_ids in prompt_token_ids
        ]

        outputs = model.generate(
            batched_prompts,
            sampling_params=sampling_params,
            use_tqdm=verbose_generation,
        )

        logprobs_dicts = [outputs[i].outputs[0].logprobs for i in range(len(outputs))]
        greedy_token_ids = [
            max(logprobs_dicts[i][0], key=lambda tid: logprobs_dicts[i][0][tid].logprob)
            for i in range(len(outputs))
        ]
        sampled_token_ids = [
            outputs[i].outputs[0].token_ids[0] for i in range(len(outputs))
        ]

        return logprobs_dicts, sampled_token_ids, greedy_token_ids

    else:
        # Only process and return a single

        outputs = model.generate(
            TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            use_tqdm=verbose_generation,
        )

        output = outputs[0].outputs[0]  # Assuming batch size & max tokens are 1

        logprobs_dict = output.logprobs  # index 0 = the single token we generated

        greedy_token_id = max(
            logprobs_dict[0], key=lambda tid: logprobs_dict[0][tid].logprob
        )
        sampled_token_id = output.token_ids[
            0
        ]  # index 0 = the single token we generated

        return logprobs_dict, sampled_token_id, greedy_token_id


def create_logprobs_dict(
    logprobs: dict[int, any],
    prompt_token_ids: list[int],
    max_token_length: int,
    model: LLM,
    sampling_params: SamplingParams,
    tokenizer: any,
    EOS_TOKEN_ID: str,
) -> dict[int, dict[str, any]]:
    """
    # TODO
    """

    extracted_logprobs = []

    generated_logprob = logprobs[0]  # index 0 = the single token we generated

    # extract the top k WORDS (not just tokens)
    for token_id, token_logprob in generated_logprob.items():
        # For each top k token for this single generated token
        full_word_ids = [token_id]  # generate first segment
        # generate the full word

        sum_logprob = token_logprob.logprob

        # If the initial token is EOS, don't try to continue generating.
        if token_id == EOS_TOKEN_ID:
            extracted_logprobs.append(
                {
                    "logprob": sum_logprob,
                    "word": tokenizer.decode(full_word_ids),
                    "ids": full_word_ids,
                }
            )
            continue

        # append the generated token to the prompt to generate the continuation of the word
        custom_prompt_token_ids = prompt_token_ids + [token_id]

        token_len = 1
        # timer_start = time.time()
        # print("------Beginning to generate for each token to explore if word or subword")
        while token_len < max_token_length:
            # Generate until we hit a space, or EOS
            next_token_logprobs_dict, next_token_sampled_id, next_token_greedy_id = (
                generate_token(custom_prompt_token_ids, model, sampling_params)
            )
            # TODO make this into its own bool hyperparameter
            next_token_id_to_use = next_token_greedy_id  # switch this to switch between partial greedy completion or partial sampling completion

            if next_token_id_to_use == EOS_TOKEN_ID:
                # full_word_ids.append(next_token_id_to_use)
                break
            decoded = tokenizer.decode([next_token_id_to_use])
            if decoded.startswith(" "):
                break
            else:
                sum_logprob += next_token_logprobs_dict[0][next_token_id_to_use].logprob
                full_word_ids.append(next_token_id_to_use)
                custom_prompt_token_ids.append(next_token_id_to_use)
                token_len += 1

        # timer_end = time.time()

        # print(f"------Done checking word/subword ({timer_end-timer_start})")

        extracted_logprobs.append(
            {
                "logprob": sum_logprob,
                "word": tokenizer.decode(full_word_ids),
                "ids": full_word_ids,
            }
        )

        # extracted_logprobs[token_id] = {
        #     # "logprob": token_logprob.logprob, # old, only first
        #     "logprob": sum_logprob, # new, actually the full word
        #     "word": tokenizer.decode(full_word_ids),
        # }

    return extracted_logprobs


def apply_word_penalties(
    logprobs: list[dict[str, any]] | list[list[dict[str, any]]],
    allowed_words: list[str],
    penalty: float,
    constraint_type: str,
    tokenizer: any,
    lemmatizer: object,
) -> list[dict[str, any]] | list[list[dict[str, any]]]:
    """
    Applies a `penalty` to all words which are present in `allowed_words`.

    Args:
        logprobs: Logprobs for each candidate for each beam branch.
        allowed_words: List of allowed words to compare candidates with
        penalty: Penalty to apply to logprobs if candidate not in `allowed_words`
        lemmatizer: spaCy lemmatizer object to lemmatize candidates for comparing with allowed words

    Returns:
        list[dict[str, any]] | list[list[dict[str, any]]]: Returns identical structure as inputted with logprobs modified
    """
    do_multiprocess = True if isinstance(logprobs[0], list) else False

    # Candidate example: {'logprob': -0.22042465209960938, 'word': ' found', 'ids': [1730]}

    def process_word(candidate: dict, tokenizer: any, lemmatizer: object):

        # if candidate["word"] == tokenizer.eos_token:
        #     print("FOUND A LONE EOS TOKEN")

        if tokenizer.eos_token in candidate["word"]:
            # print("Found EOS token in word, auto allow...")
            return True  # Allow EOS tokens so sequences can naturally terminate

        decontracted = contractions.fix(candidate["word"])

        # tag_cleaned = decontracted.replace(tokenizer.eos_token, "")

        removed_specials_around = re.sub(
            r"^\W+|\W+$", "", decontracted
        )  # Remove only spcial characters before and after (eg. "-don't." -> "don't")

        # # print("Started lemmatizing.")
        doc = lemmatizer(removed_specials_around)

        lemmas = [
            item.lemma_ for item in doc
        ]  # Can be several, usually 1 for the single last word

        pos = [item.pos_ for item in doc]

        # print(pos)

        excluded_classes = set(["X", "SYM", "PROPN", "NUM", "SPACE", "PUNCT"])

        exclude_indecies = [i for i, item in enumerate(pos) if item in excluded_classes]

        lemmas = [item for i, item in enumerate(lemmas) if i not in exclude_indecies]
        pos = [item for i, item in enumerate(pos) if i not in exclude_indecies]

        # print("Found lemmas:")
        # for lem in doc:
        # print(f"  {lem.lemma_}, ({lem.pos_})")

        # print(f"---{candidate["word"]}'s lemmas: {lemmas}")

        # print("Allowed Words:")
        # print(allowed_words)

        if set(lemmas) & set(
            allowed_words
        ):  # Any overlap between word lemma and allowed words
            # print("Found overlap!, Is ok.")
            # print(last_generated_word_lemmas)
            # print(self.allowed_words)
            return True
        else:
            # print("No overlap? Applying penalty")
            return False

    if do_multiprocess:
        for branch in logprobs:
            for candidate in branch:
                # print(f"Decision \"{candidate["word"]}\"")
                is_allowed = process_word(candidate, tokenizer, lemmatizer)
                # print("------Allowed") if is_allowed else print("------Denied")
                if not is_allowed:
                    if constraint_type == "soft":
                        candidate["logprob"] -= penalty
                    elif constraint_type == "hard":
                        candidate["logprob"] = float("-inf")
                    else:
                        raise TypeError(
                            f'Invalid word constraint type. "soft" | "hard", got "{constraint_type}"'
                        )

    else:
        for candidate in logprobs:
            is_allowed = process_word(candidate, tokenizer, lemmatizer)
            if not is_allowed:
                if constraint_type == "soft":
                    candidate["logprob"] -= penalty
                elif constraint_type == "hard":
                    candidate["logprob"] = float("-inf")
                else:
                    raise TypeError(
                        f'Invalid word constraint type. "soft" | "hard", got "{constraint_type}"'
                    )

    return logprobs


def find_optimal_beams(
    logprobs_dict: dict[int, any] | list[dict[int, any]],
    beam_tree: BeamSearch,
    prompt_token_ids: list[int],
    allowed_words: list[str],
    model: LLM,
    sampling_params: SamplingParams,
    tokenizer: any,
    lemmatizer: object,
    EOS_TOKEN_ID: str,
    max_possible_token_length: int,
    use_word_constraint: bool = False,
    word_soft_constraint_penalty: float = 2.5,
    word_constraint_type: str = "hard",
):
    """
    Using a single token output logprobs, finds the optimal beams given a beam tree object.
    # TODO
    Args:
        None

    Returns:
        list[int]: List of token IDs representing the last generated token for the optimal beams.
    """

    extracted_logprobs = {}

    # Do some logic of removing words that are not in the vocab.
    if beam_tree.initialized == False:
        # initialize

        # Example for 2 beams:
        # [
        #   {'logprob': -1.1990221738815308, 'word': 'John', 'ids': [13079]},
        #   {'logprob': -1.3396471738815308, 'word': 'The', 'ids': [785]}
        # ]
        # timer_start = time.time()
        # print("----Making Logprobs Object...")
        extracted_logprobs = create_logprobs_dict(
            logprobs_dict,
            prompt_token_ids,
            max_possible_token_length,
            model,
            sampling_params,
            tokenizer,
            EOS_TOKEN_ID,
        )

        if use_word_constraint:
            logprobs_with_applied_penalties = apply_word_penalties(
                extracted_logprobs,
                allowed_words,
                word_soft_constraint_penalty,
                word_constraint_type,
                tokenizer,
                lemmatizer,
            )
        else:
            logprobs_with_applied_penalties = extracted_logprobs
        # timer_end = time.time()
        # print(f"----Logprobs Object Done. ({timer_end - timer_start})")

        # timer_start = time.time()
        # print("----Updating Beam Tree...")
        beams = beam_tree.update(logprobs_with_applied_penalties)
        # timer_end = time.time()
        # print(f"----Beam Tree Updated. ({timer_end - timer_start})")

    else:
        logprobs_dicts = logprobs_dict  # Now a list of dicts, one for each beam
        generated_token_ids = prompt_token_ids  # Now list of prompt + previously generated tokens for each beam

        extracted_logprobs = []

        # Example 2 beams:
        # [
        #   [
        #     {'logprob': -0.22042465209960938, 'word': ' found', 'ids': [1730]},
        #     {'logprob': -2.4547996520996094, 'word': ' lost', 'ids': [5558]}
        #   ],
        #   [
        #     {'logprob': -1.1107125282287598, 'word': ' guy', 'ids': [7412]},
        #     {'logprob': -1.3607125282287598, 'word': ' man', 'ids': [883]}
        #   ]
        # ]
        # timer_start = time.time()
        # print("----Making Logprobs Object...")
        for beam_i in range(beam_tree.beam_size):
            extracted_logprobs.append(
                create_logprobs_dict(
                    logprobs_dicts[beam_i],
                    generated_token_ids[beam_i],
                    max_possible_token_length,
                    model,
                    sampling_params,
                    tokenizer,
                    EOS_TOKEN_ID,
                )
            )

        if use_word_constraint:
            logprobs_with_applied_penalties = apply_word_penalties(
                extracted_logprobs,
                allowed_words,
                word_soft_constraint_penalty,
                word_constraint_type,
                tokenizer,
                lemmatizer,
            )
        else:
            logprobs_with_applied_penalties = extracted_logprobs

        # timer_end = time.time()
        # print(f"----Logprobs Object Done. ({timer_end - timer_start})")

        # timer_start = time.time()
        # print("----Updating Beam Tree...")
        beams = beam_tree.update(logprobs_with_applied_penalties)
        # timer_end = time.time()
        # print(f"----Beam Tree Updated. ({timer_end - timer_start})")

    # Extract the last token of the beams which is the newly generated token

    return beams


def build_chat_with_token_ids(
    system_prompt: str, user_prompt: str, tokenizer: any
) -> list[int]:
    """
    Builds initial chat history with prompt converted to token ids to start a new conversation history with vLLM.

    Args:
        system_prompt: The system prompt to set the behavior of the model.
        user_prompt: The user prompt containing the prompt to ask the model.
        tokenizer: The tokenizer to convert the prompts to token ids (using for example vLLM `use_tokenizer`)

    Returns:
        list[int]: The list of token IDs representing the initial chat history.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # apply_chat_template adds the <|begin_of_text|>, roles, etc.
    token_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # appends the assistant turn opener
        return_tensors=None,  # returns a plain Python list
    )
    return token_ids
