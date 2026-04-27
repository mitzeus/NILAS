import os
import json


def get_generated_outputs(
    generator_name: str, language: str, path: str = "chats/self-iteration"
):
    folder_path = os.path.join(path, generator_name, language)

    texts = []

    for folder, subfolders, files in os.walk(folder_path):
        for file in files:
            if file != "best.json":
                continue

            file_path = os.path.join(folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts.append(data["generated_text"])

    return texts
