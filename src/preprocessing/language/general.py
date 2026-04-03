import pandas as pd
import numpy as np
import os

from src.preprocessing.probabilities import class_prior, hamilton


def convert_frequency_to_WPM(frequencies: np.array[int], round_to: int = 2):
    total_size = frequencies.sum()

    frequencies_wpm = np.round((frequencies / total_size) * 1000000, round_to)

    return frequencies_wpm


def create_sorted_flashcard_set(
    data: pd.DataFrame,
    data_columns: list[str],
    pos_str: str,
    frequency_str: str,
    rank_by: str,
    lang: str,
    target_columns: list[str] = None,
    drop_pos: list[str] = None,
    limit: int = 100,
):
    """
    Creates the final flashcard set for a language by picking
    top candidates for each PoS tag/grammar type corresponding
    to the distribution int the data.

    Args:
        data: Full dataset of lemmas
        data_columns: List of strings representing which dataframe columns to keep in the final output
        lang: Name of the language to process. Used for filename generation.
        target_columns: List of strings to rename `data_columns`. Name is assigned index-wise.
        pos_str: Column name for PoS tag/grammar type.
        frequency_str: Column name for word frequencies.
        rank_by: Ranks the final flashcard set in descending order using this column.
        drop_pos: Removes the following PoS tags (useful for removing for example punctuations)
        limit: Limits total size of final flashcard set.

    Returns:
        DataFrame: Final Flashcard set
    """
    if len(data_columns) != len(target_columns):
        raise TypeError("data_columns and target_columns are different lengths")

    # Build DataFrame with new columns
    df = pd.DataFrame(columns=target_columns)

    restructured_data = pd.DataFrame(columns=target_columns)
    for i in range(len(target_columns)):
        restructured_data[target_columns[i]] = data[data_columns[i]]

    restructured_data = restructured_data[~restructured_data[pos_str].isin(drop_pos)]

    data = restructured_data

    # # Calculate distribution
    percentages, class_prior_fig = class_prior(
        data, pos_col=pos_str, freq_col=frequency_str
    )

    # # Pick top words
    discrete_amounts, hamilton_fig = hamilton(percentages=percentages, limit=limit)

    # Put together df
    candidate_indexes = []

    for word_class in discrete_amounts.index:
        classwise_subset = data[data[pos_str] == word_class]
        classwise_subset = classwise_subset.sort_values(
            by=frequency_str, ascending=False
        )  # sort to get top words
        candidate_indexes.append(
            classwise_subset.head(discrete_amounts.loc[word_class]).index
        )
        # print(f"Added {word_class} with {discrete_amounts.loc[word_class]} entries.")

    candidate_indexes = np.concatenate(candidate_indexes)

    df = data.loc[candidate_indexes]

    df = df.sort_values(by=rank_by, ascending=False)

    # write to 3-final
    target_dir = "data/3-final/"
    df.to_csv(os.path.join(target_dir, f"{lang}{limit}.csv"), index=False)

    # return df, prob dist. fig., hamilton fig.
    return df, class_prior_fig, hamilton_fig
