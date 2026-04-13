from openai import OpenAI
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))


def ask_chat_conv(
    prompt: str,
    system_prompt: str,
    flashcards: str,
    keep_history: bool = True,
    target_language: str = None,
    preferred_language: str = None,
    level: str = None,
):

    # Build the engineered prompt

    engineered_prompt = f"""

    {system_prompt}

    The sentence is:
    "{prompt}"

    Flashcard List:
    {flashcards}

"""

    response = client.responses.create(
        model="gpt-5.2",
        input=engineered_prompt,
        temperature=0.7,
        max_output_tokens=1000,
        store=False,
    )

    print(response.output_text)


# TODO make a one ask model with no history
def ask_chat_once(prompt: str) -> str:
    pass

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
    )

    print(response)
