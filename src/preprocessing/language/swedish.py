import pandas as pd


def remove_and_merge_pos(data: pd.DataFrame, clump_word_classes: dict):
    """
    Removes and merges PoS tags/grammar types together. It's used for example in case of duplicate PoS tags
    or PoS tags that would be better off classified as one.

    Args:
        data: DataFrame to find and merge PoS tags on.
        clump_word_classes: A dict containing {string: list[string]} pairs. List of strings will be merged to become the key.

    Returns:
        DataFrame: Clumped DataFrame based on `clump_word_classes`.

    """
    # clump similar word classes together, (e.g. all nouns together, verbs together etc.)
    word_types = data["Word classes"]

    for (
        word_type,
        word_class_list,
    ) in clump_word_classes.items():  # clumping word classes together
        word_types = word_types.replace(word_class_list, word_type)

    swedish_clumped = data.copy()
    swedish_clumped["Word classes"] = word_types

    # remove extra info from word itself
    swedish_clumped["Swedish items for translation"] = swedish_clumped[
        "Swedish items for translation"
    ].str.replace(r"\s*\([^)]*\)", "", regex=True)

    return swedish_clumped
