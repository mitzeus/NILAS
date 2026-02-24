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
        User Question:
        {prompt}

        User Flashcards (Known Words):
        {flashcards}

        Target Language: {target_language}
        User Level: {level}
        Preferred Language: {preferred_language}
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

    # append prompt and system_prompt to history
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": engineered_prompt},
    ]

    # Keep the conversation going
    while True:
        response = client.responses.create(
            model="gpt-5.2",
            input=history,
            temperature=0.7,
            max_output_tokens=1000,
            store=False,
        )

        print(response.output_text)

        if keep_history:
            history += [
                {"role": el.role, "content": el.content} for el in response.output
            ]
        else:
            history = [{"role": "system", "content": system_prompt}]

        new_prompt = input("> ")

        engineered_new_prompt = f"""
            User Question:
            {new_prompt}

            User Flashcards (Known Words):
            {flashcards}

            Target Language: {target_language}
            User Level: {level}
            Preferred Language: {preferred_language}
        """.strip()
        engineered_new_prompt += (
            f"\n Target Language: {target_language}" if target_language != None else ""
        )
        engineered_new_prompt += f"\n User Level: {level}" if level != None else ""
        engineered_new_prompt += (
            f"\n Preferred Language: {preferred_language}"
            if preferred_language != None
            else ""
        )

        history.append({"role": "user", "content": engineered_new_prompt})


# TODO make a one ask model with no history
def ask_chat_once(prompt: str) -> str:
    pass

    response = client.responses.create(
        model="gpt-5.2",
        input=prompt,
    )

    print(response)
