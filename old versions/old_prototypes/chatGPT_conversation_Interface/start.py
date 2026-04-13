from prompts import (
    SYSTEM_PROMPT_STRING,
    SYSTEM_PROMPT_STRING_NEW,
)
from flashcards import spanish50_limited as FLASHCARD_PROMPT_STRING
from models import ask_chat_conv

# TODO "- **Estoy en escuela.** (more natural: “en la escuela,” but **la** is new, so we’ll keep it simple)", prioritize natural answer when user specifically asks for it,
# only then CI can be relatexed

initial_prompt = "I want to practice simple conversation in Spanish."

print(initial_prompt)

ask_chat_conv(
    prompt=initial_prompt,
    system_prompt=SYSTEM_PROMPT_STRING_NEW,
    flashcards=FLASHCARD_PROMPT_STRING,
    keep_history=True,
    target_language=None,
    preferred_language=None,
    level=None,
)
