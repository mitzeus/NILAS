from prompts import (
    SYSTEM_PROMPT_STRING,
)
from flashcards import spanish50_limited as FLASHCARD_PROMPT_STRING
from models import ask_chat_conv

# TODO "- **Estoy en escuela.** (more natural: “en la escuela,” but **la** is new, so we’ll keep it simple)", prioritize natural answer when user specifically asks for it,
# only then CI can be relatexed

evaluate_sentence = "Hola!   ¿Cómo estás? Hoy voy a trabajar en una escuela. Yo soy feo y grande. tú hablas espanol?"

print(evaluate_sentence)

ask_chat_conv(
    prompt=evaluate_sentence,
    system_prompt=SYSTEM_PROMPT_STRING,
    flashcards=FLASHCARD_PROMPT_STRING,
    keep_history=True,
    target_language=None,
    preferred_language=None,
    level=None,
)
