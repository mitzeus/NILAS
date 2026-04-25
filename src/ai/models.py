import numpy as np

from src.ai.beam_search import BeamSearch
from src.ai.tools import (
    build_chat_with_token_ids,
    generate_token,
    find_optimal_beams,
    get_max_token_length_of_vocab,
    process_vocab_to_prompt,
)
from vllm.inputs import TokensPrompt


class Custom_vLLM:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        lemmatizer: object,
        allowed_words: list[str],
        beam_size: int = 1,
        word_soft_constraint_penalty: float = 2.5,
        alpha: float = 0.6,
    ):
        self.output = ""
        self.perplexity = None

        self.beam_average_logprobs = None

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.lemmatizer = lemmatizer
        self.allowed_words = allowed_words
        self.beam_size = beam_size
        self.word_soft_constraint_penalty = word_soft_constraint_penalty
        self.alpha = alpha

        self.beam_tree = BeamSearch(
            beam_size=self.beam_size,
            allowed_words=self.allowed_words,
            tokenizer=self.tokenizer,
            allowed_word_penalty=self.word_soft_constraint_penalty,
            alpha=self.alpha,
        )

    def _calculate_perplexity(self):
        """
        Calculates perplexity, either for the best selected beam if finished, or for each of the non-finished beam candidates
        """
        if isinstance(self.beam_average_logprobs, list):
            self.perplexity = [np.exp(-logprob) for logprob in self.beam_logprobs]
        elif isinstance(self.beam_average_logprobs, float):
            self.perplexity = np.exp(-self.beam_average_logprobs)
        else:
            raise TypeError(
                f"beam_average_logprob is invalid type. Expected `list` | `int`, got {type(self.beam_average_logprobs)}"
            )

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        sampling_params: object,
        max_sequence_length: int = 200,
        verbose: str | None = "sequence",
        use_word_constraint=False,
        word_constraint_type: str = "hard",
        prompt_allowed_words: bool = False,
    ):
        system_prompt = process_vocab_to_prompt(
            system_prompt, self.allowed_words, prompt_allowed_words
        )

        max_possible_token_length_of_vocab = get_max_token_length_of_vocab(
            self.allowed_words, self.tokenizer
        )

        original_token_ids = build_chat_with_token_ids(
            system_prompt, user_prompt, self.tokenizer
        )
        token_ids_for_beams = [
            [] for _ in range(self.beam_tree.beam_size)
        ]  # Use this to keep track of token IDs for each beam
        prompts_token_ids = [
            original_token_ids.copy() for _ in range(self.beam_tree.beam_size)
        ]  # Use this to build original + generated token

        for seq_step in range(max_sequence_length):
            if verbose == "full" or verbose == "sequence":
                print(f"Sequence step: {seq_step + 1}")
            # Generate one token per seq step (in the beam tree it is processing full breath and each token is one step in depth)

            if self.beam_tree.initialized == False:
                # Generate the beam root
                (
                    print(f"Generating for {self.beam_tree.beam_size} beams.")
                    if verbose == "full"
                    else None
                )

                logprobs_dict, sampled_token_id, greedy_token_id = generate_token(
                    original_token_ids, self.model, sampling_params
                )

                (
                    print(f"Generated. Now finding optimal beams...")
                    if verbose == "full"
                    else None
                )

                beam_objects = find_optimal_beams(
                    logprobs_dict,
                    self.beam_tree,
                    original_token_ids,
                    self.allowed_words,
                    self.model,
                    sampling_params,
                    self.tokenizer,
                    self.lemmatizer,
                    self.eos_token_id,
                    max_possible_token_length=max_possible_token_length_of_vocab,
                    use_word_constraint=use_word_constraint,
                    word_soft_constraint_penalty=self.word_soft_constraint_penalty,
                    word_constraint_type=word_constraint_type,
                )

                print("Found optimal beams.") if verbose == "full" else None

                for i, beam in enumerate(beam_objects):
                    prompts_token_ids[i] = (
                        original_token_ids.copy() + beam.ids
                    )  # add the last chosen token ids to the respective generated token id list (original + generated)

                (
                    print(
                        "Now appended chosen tokens. Continuing to next sequence step...\n"
                    )
                    if verbose == "full"
                    else None
                )

            else:
                # Continue generating for N beams
                (
                    print(f"Generating for {self.beam_tree.beam_size} beams.")
                    if verbose == "full"
                    else None
                )

                logprobs_dicts, sampled_token_ids, greedy_token_ids = generate_token(
                    prompts_token_ids,
                    self.model,
                    sampling_params,  # now inputting list of prompt + generated token ids for each beam
                )

                (
                    print(f"Generated. Now finding optimal beams...")
                    if verbose == "full"
                    else None
                )

                beam_objects = (
                    find_optimal_beams(  # will find best beams over all branches
                        logprobs_dicts,
                        self.beam_tree,
                        prompts_token_ids,
                        self.allowed_words,
                        self.model,
                        sampling_params,
                        self.tokenizer,
                        self.lemmatizer,
                        self.eos_token_id,
                        max_possible_token_length=max_possible_token_length_of_vocab,
                        use_word_constraint=use_word_constraint,
                        word_soft_constraint_penalty=self.word_soft_constraint_penalty,
                        word_constraint_type=word_constraint_type,
                    )
                )

                if self.beam_tree.finished:
                    print("Fully finished.")
                    self.output = self.tokenizer.decode(
                        self.beam_tree.best_beam.ids, skip_special_tokens=True
                    )
                    self.beam_average_logprobs = (
                        self.beam_tree.best_beam.logprob
                    ) / len(self.beam_tree.best_beam.ids)
                    break

                print("Found optimal beams.") if verbose == "full" else None

                for i, beam in enumerate(beam_objects):
                    prompts_token_ids[i] = original_token_ids.copy() + beam.ids

                (
                    print(
                        "Now appended chosen tokens. Continuing to next sequence step...\n"
                    )
                    if verbose == "full"
                    else None
                )

        if not self.beam_tree.finished:
            print("Did not finish before seqence maximum.")
            self.output = [obj.sequence for obj in self.beam_tree.beams]
            self.beam_average_logprobs = [
                obj.logprob / len(obj.ids) for obj in self.beam_tree.beams
            ]

        self._calculate_perplexity()

        return self.output, self.perplexity


class Vanilla_vLLM:
    def __init__(
        self,
        model: object,
        tokenizer: object,
        allowed_words: list[str],
    ):
        self.output = ""
        self.perplexity = None
        self.sequence_logprobs = []

        self.model = model
        self.tokenizer = tokenizer
        self.allowed_words = allowed_words

    def _calculate_perplexity(self):
        self.perplexity = np.exp(-np.mean(self.sequence_logprobs))

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        sampling_params: object,
        max_sequence_length: int = 200,
        prompt_allowed_words: bool = False,
        verbose: bool = False,
    ):

        system_prompt = process_vocab_to_prompt(
            system_prompt, self.allowed_words, prompt_allowed_words
        )

        original_token_ids = build_chat_with_token_ids(
            system_prompt, user_prompt, self.tokenizer
        )

        outputs = self.model.generate(
            TokensPrompt(prompt_token_ids=original_token_ids),
            sampling_params=sampling_params,
            use_tqdm=verbose,
        )

        self.output = outputs[0].outputs[0].text
        for token in outputs[0].outputs[0].logprobs:
            for chosen in token.values():
                self.sequence_logprobs.append(chosen.logprob)

        self._calculate_perplexity()

        return self.output, self.perplexity


class Vanilla_ChatGPT:
    def __init__(self, client: object, model: str, allowed_words: list[str]):
        self.output = ""
        self.perplexity = None
        self.sequence_logprobs = None

        self.client = client
        self.model = model
        self.allowed_words = allowed_words

    def _calculate_perplexity(self):
        self.perplexity = np.exp(-np.mean(self.sequence_logprobs))

    def __call__(
        self,
        system_prompt: str,
        user_prompt: str,
        max_sequence_length: int = 200,
        prompt_allowed_words: bool = False,
    ):

        system_prompt = process_vocab_to_prompt(
            system_prompt, self.allowed_words, prompt_allowed_words
        )

        # print(system_prompt)

        prompt = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ]

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=max_sequence_length,
            include=["message.output_text.logprobs"],
        )

        self.output = response.output_text
        self.sequence_logprobs = [
            token_info.logprob for token_info in response.output[0].content[0].logprobs
        ]

        self._calculate_perplexity()  # do this before finishing

        return self.output, self.perplexity
