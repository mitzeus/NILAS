import pandas as pd
import json


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
                    "content": [{"type": "input_text", "text": response.output_text}],
                }
            ]
        else:
            self.history = [
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": response.output_text}],
                }
            ]

        if self.save_history_to_file:
            self._save_history()

        return response.output_text


class Corrector:
    def __init__(self):
        pass
