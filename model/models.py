import pandas as pd
import json
import spacy
import re


class Prompt_Preprocessor:
    def __init__(self):
        pass

    def __call__(
        self,
        prompt: str,
        flashcards: pd.DataFrame,
        target_language: str = None,
        preferred_language: str = None,
        level: str = None,
    ) -> str:
        engineered_prompt = f"""
        User Question:
            {prompt}

            User Flashcards (Known Words):
            {flashcards}


        """.strip()

        engineered_prompt += (
            f"\n Target Language: {target_language}" if target_language != None else ""
        )
        engineered_prompt += f"\n User Level: {level}" if level != None else ""
        engineered_prompt += (
            f"\n Preferred Language: {preferred_language}"
            if preferred_language != None
            else ""
        )

        return engineered_prompt


process_prompt = Prompt_Preprocessor()


class Conversation_Model:
    def __init__(
        self,
        system_message: str,
        model_client: object,
        model_name: str,
        temperature: float = 0.3,
        max_output_tokens: int = 1000,
        store: bool = False,
        keep_history: bool = True,
        save_history_to_file: bool = True,
    ):
        self.model_client = model_client
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.store = store
        self.keep_history = keep_history
        self.save_history_to_file = save_history_to_file
        if self.save_history_to_file:
            self.file_save_name = (
                f"chats/chat_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        self.system_message = system_message

        self.flashcards = None
        self.history = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self.system_message}],
            }
        ]

    def _save_history(self):
        with open(self.file_save_name, "w") as f:
            json_history = json.dumps(self.history, indent=4)
            f.write(json_history)

    def import_word_library(self, flashcards: pd.DataFrame):
        # Takes dataframe with columns: [freq	level	affix	word]

        flashcards_string = """"""

        for _, row in flashcards.iterrows():
            flashcards_string += f"{row["word"]}\n"

        self.flashcards = flashcards_string

    def determine_level(self) -> str:
        # TODO: implement method to determine user level based on flashcards CEFR level and amount of words in flashcard library
        pass

    def ask(
        self,
        prompt: str,
        target_language: str = None,
        preferred_language: str = None,
        level: str = None,
    ) -> str:
        if self.flashcards == None:
            print(
                "WARNING: No word library has been imported. No flashcard limitations are set."
            )

        preprocessed_prompt = process_prompt(
            prompt=prompt,
            flashcards=self.flashcards,
            target_language=target_language,
            preferred_language=preferred_language,
            level=level,
        )

        if self.keep_history:
            self.history += [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": preprocessed_prompt}],
                }
            ]
        else:
            self.history = [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_message}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": preprocessed_prompt}],
                },
            ]

        response = self.model_client.responses.create(
            model=self.model_name,
            input=self.history,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            store=self.store,
        )

        if self.keep_history:
            self.history += [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response.output_text}],
                }
            ]
        else:
            self.history = [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": response.output_text}],
                }
            ]

        if self.save_history_to_file:
            self._save_history()

        return response.output_text


class Conversation:
    def __init__(
        self,
        id: str,
        question_id: str,
        nr_vocab: int,
        question: str,
        response: str,
        word_limits: str = None,
    ):
        self.id = id
        self.question_id = question_id
        self.nr_vocab = nr_vocab
        self.question = question
        self.response = response
        self.word_limits = word_limits

        self.lexical = self.Lexical(self)
        self.naturalness = self.Naturalness(self)

    class Lexical:
        def __init__(self, parent):
            self.parent = parent

    class Naturalness:
        def __init__(self, parent):
            self.parent = parent


class Corrector:
    """
    The corrector is a module that implements improvement and evaluation steps.
    It handles checking measurements of the output with respect to lexical con-
    straints and naturalness, then sends feedback back to the model for iterative
    improvement.
    """

    def __init__(self):
        """
        ## Args:
            None

        ## Returns:
            None
        """
        # Define subcomponents of corrector
        self.lexical = self.Lexical(self)
        self.naturalness = self.Naturalness(self)

        # Attributes
        self.conversation_data = None
        self.flashcards = None

    def fit(self, conversation_data: list[dict], flashcards: list[str] = None):
        """
        Fits LLM conversation history to corrector model, enabling usage of improvement and evaluation methods.

        ## Args:
            `conversation_data (list[dict])`: Conversation history from an API model.

            `flashcards (str)`: String of words separated by `\\n`

        ## Returns:
            None
        """
        self.conversation_data = conversation_data
        self.flashcards = flashcards

    class Lexical:
        """
        All methods related to lexical constraint processing.
        """

        def __init__(self, parent):
            self.parent = parent

        def llm_classification(
            self, model_client: object, model_name: str, system_message: str
        ) -> str:
            """
            Classifies each word in a string as new or old given `self.flashcards` using a separate LLM model.

            All results DataFrames are appended to Conversation object: `conversation.lexical.llm_classification`.

            ## Args:
                `model_client (object)`: Object for model client to use
                `model_name (str)`: Model name to use in the client
                `system_message (str)`: System Message used for model instructions

            ## Returns:
                `list[pd.DataFrame]`: List of Dataframe with columns "word" and "is_new" for each word in the string

            """
            conversations = self.parent.conversation_data

            list_of_dfs = []

            for conversation in conversations:

                response = conversation.response.split("/?VOCABULARY?/")
                response = response[0]

                full_prompt = f"""
                The sentence is:
                "{response}"

                Flashcard List:
                {self.parent.flashcards}
                """

                rated_response = model_client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_message}],
                        },
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": full_prompt}],
                        },
                    ],
                    temperature=0.1,
                    max_output_tokens=10000,
                    store=False,
                )

                # print(rated_response.output_text)

                # assumes this version:
                # Varje,0
                # dag,0
                # vaknar,0

                word_status_pairs = rated_response.output_text.split("\n")

                df = pd.DataFrame(columns=["word", "is_new"])
                words = []
                is_news = []

                for word_status_pair in word_status_pairs:
                    word_status_pair_splitted = word_status_pair.split(
                        ","
                    )  # split word and status

                    if len(word_status_pair_splitted) != 2:
                        continue

                    words.append(word_status_pair_splitted[0])
                    is_news.append(int(word_status_pair_splitted[1]))

                df["word"] = words
                df["is_new"] = is_news

                conversation.lexical.llm_classification = df

                list_of_dfs.append(df)

            return list_of_dfs

        def raw_checking(self):
            """
            Processes model output using traditional NLP methods
            and compares the used vocabulary with the allowed vocabulary
            lists. Each word is then marked according to if it is an existing or new
            word and checks both if all new words are marked correctly and if the
            output followed CI constraint

            ## Args:
                None

            ## Returns:
                `pd.DataFrame`
            """
            # TODO implement checking using traditional methods

            # Lemmatization and NLP proecessing from scratch
            # SpaCy is a popular library for lemmatization

            # There is also a version using spaCy to train yourself, but as we have POS tags, that's what will be used
            # SpaCy also has support for Spanish, Swedish and Korean

            # Available Models
            # Korean: ko_core_news_lg (large),  ko_core_news_sm (small)
            # Spanish: es_dep_news_trf (large), es_core_news_sm (small)
            # Swedish: sv_core_news_lg (large), sv_core_news_sm (small)

            conversations = self.parent.conversation_data

            list_of_dfs = []

            for conversation in conversations:

                # Extract only text
                text = conversation.response.split("/?VOCABULARY?/")
                text = text[0]

                # preprocess
                text = text.lower()
                text = re.sub(r"[^\w\s]|\n", "", text)  # removes special letters

                nlp = spacy.load("sv_core_news_lg")  # for Swedish

                word_list = []
                lemma_list = []
                score_list = []

                doc = nlp(text)
                # print(doc.text)
                for token in doc:
                    # print(token.text, "->", token.lemma_)
                    lemma = token.lemma_

                    if token.lemma_ in self.parent.flashcards:
                        score_list.append(0)
                    else:
                        score_list.append(1)

                    word_list.append(token.text)
                    lemma_list.append(lemma)

                df = pd.DataFrame(columns=["word", "lemma", "score"])
                df["word"] = word_list
                df["lemma"] = lemma_list
                df["score"] = score_list

                conversation.lexical.raw_checking = df

                list_of_dfs.append(df)

            return list_of_dfs

    class Naturalness:
        """
        All methods related to Naturalness processing.
        """

        def __init__(self, parent):
            self.parent = parent

        def perplexity(self):
            """
            Computes perplexity given a word sequence.

            ## Args:
                None

            ## Returns:
                `float`
            """
            # TODO implement perplexity
            pass

        def llmaaj(self, model: object, args: dict):
            """
            Lets other LLMs judge and rate areas of the model
            output and propose improvements. Several different LLMs would allow
            for more advanced reasoning and result in a more nuanced final rating
            and critics.

            ## Args:
                `model (object)`: Object for model to use
                `args (dict)`: arguments that will be passed to model


            ## Returns:
                `pd.DataFrame`: Ratings of each category

                `str`: Prompt for improving model output
            """
            # TODO Implement LLM as a Judge
            pass

        def human_compare(self, human_corpus: str):
            """
            Calculates distributions of key measurements of
            both the model output and a human-written corpus and com-
            pares how similar the distributions are using KL-Divergence

            ## Args:
                `human_corpus (str)`: Human-written corpus/text that the method will compare the generated output to


            ## Returns:
                `pd.DataFrame`: DataFrame of Distribution distances based on category

            """
            # TODO Implement Text Distribution Comparison using human corpus
            pass
