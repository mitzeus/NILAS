import pandas as pd
import numpy as np
from numpy.typing import NDArray
import os

from src.preprocessing.probabilities import class_prior, hamilton


def convert_frequency_to_WPM(frequencies: NDArray[np.int_], round_to: int = 2):
    """
    Converts raw frequency to Words per Million (WPM) metric.

    Args:
        frequencies: List of raw frequencies.
        round_to: Rounds WPM values to N fractions.

    Returns:
        list: List of frequencies converted to WPM
    """
    total_size = frequencies.sum()

    frequencies_wpm = np.round((frequencies / total_size) * 1000000, round_to)

    return frequencies_wpm


def _redistribute_hamilton_shortfall(
    discrete_amounts: pd.Series, data: pd.DataFrame, pos_str: str
) -> pd.Series:
    """
    Ensure discrete allocations do not exceed available items per PoS.

    If a PoS class has fewer available items than Hamilton requested,
    the unfulfilled amount is redistributed among other classes that still
    have available capacity.
    """
    available_counts = data[pos_str].value_counts()
    adjusted = discrete_amounts.copy()
    deficit = 0

    for word_class, amount in adjusted.items():
        available = int(available_counts.get(word_class, 0))
        if amount > available:
            deficit += amount - available
            adjusted.loc[word_class] = available

    if deficit <= 0:
        return adjusted

    capacities = (
        available_counts.reindex(adjusted.index).fillna(0).astype(int) - adjusted
    )
    capacities = capacities[capacities > 0]
    if capacities.empty:
        return adjusted

    while deficit > 0 and not capacities.empty:
        # Allocate to the class with the most remaining capacity first.
        largest_capacity_class = capacities.idxmax()
        adjusted.loc[largest_capacity_class] += 1
        capacities.loc[largest_capacity_class] -= 1
        deficit -= 1
        if capacities.loc[largest_capacity_class] <= 0:
            capacities = capacities.drop(largest_capacity_class)

    return adjusted


def create_sorted_flashcard_set(
    data: pd.DataFrame,
    data_columns: list[str],
    pos_str: str,
    frequency_str: str,
    rank_by: str,
    lang: str,
    target_columns: list[str] = None,
    drop_pos: list[str] = [],
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
    discrete_amounts, hamilton_fig = hamilton(
        percentages=percentages, limit=limit, lang=lang
    )
    discrete_amounts = _redistribute_hamilton_shortfall(discrete_amounts, data, pos_str)

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


def sample_word_sublist(df: pd.DataFrame, vocab_size: int) -> pd.DataFrame:
    required_cols = {"word", "pos", "frequency"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns {required_cols}")

    if len(df) < vocab_size:
        raise ValueError("Dataframe words are less than proposed vocab size extraction")

    if vocab_size <= 0:
        return df.iloc[0:0].copy()

    pos_counts = df["pos"].value_counts()
    total_count = len(df)

    quotas = (pos_counts / total_count) * vocab_size

    base_alloc = np.floor(quotas).astype(int)

    remainder = quotas - base_alloc
    remaining = vocab_size - base_alloc.sum()

    if remaining > 0:
        extra = remainder.sort_values(ascending=False).index[:remaining]
        for pos in extra:
            base_alloc[pos] += 1

    selected_rows = []

    for pos, n in base_alloc.items():
        subset = df[df["pos"] == pos]
        subset_sorted = subset.sort_values(by="frequency", ascending=False)
        selected_rows.append(subset_sorted.head(n))

    result = pd.concat(selected_rows)

    if len(result) != vocab_size:
        raise RuntimeError(
            f"Selection failed: expected {vocab_size}, got {len(result)}"
        )

    return result.reset_index(drop=True)
