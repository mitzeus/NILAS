import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def class_prior(data: pd.DataFrame, pos_col: str, freq_col: str):
    """
    # TODO
    """
    # calculate percentages
    total_freq = data[freq_col].sum()

    percentages = pd.DataFrame(columns=["pos", "percentages"])

    for word_type in data[pos_col].unique():
        word_type_freq = data[data[pos_col] == word_type][freq_col].sum()

        percentage = (word_type_freq / total_freq) * 100
        percentages = pd.concat(
            [
                percentages,
                pd.DataFrame([{"pos": word_type, "percentages": percentage}]),
            ],
            ignore_index=True,
        )

    # create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.bar(percentages["pos"], percentages["percentages"])
    # plt.figsize = (10, 6)
    ax.set_xlabel("Word Classes")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Probability Distribution per Grammar Type")
    ax.tick_params(axis="x", rotation=45)
    plt.close(fig)

    return percentages, (fig, ax)


def hamilton(percentages: pd.DataFrame, limit: int):
    """
    # TODO
     Calculates discrete amounts based on percentages (where each category gets at least 1 entry)
    """

    pos, percentages = percentages["pos"], percentages["percentages"]

    divisor = limit - len(pos)

    discrete_percentages = pd.DataFrame(columns=["pos", "discrete_percentages"])

    percentages = percentages.to_numpy()
    discrete_list = np.floor(
        ((percentages / 100) * divisor) + 1
    )  # gets the floored version of amounts
    fractional_parts = (
        ((percentages / 100) * divisor) + 1 - discrete_list
    )  # saves unused fractions that was removed with flooring

    leftover_amount = int(
        limit - discrete_list.sum()
    )  # how many entries left to distribute

    fractional_parts = pd.Series(fractional_parts, index=pos).sort_values(
        ascending=False
    )  # link it to word classes

    discrete_amounts = pd.Series(discrete_list, index=pos, dtype=int)

    for i in range(leftover_amount):
        discrete_amounts.loc[
            fractional_parts.index[i]
        ] += 1  # distributes last leftovers to top fractions

    # print(
    #     f"Percentages converted to discrete amounts (with at least 1 entry for each category): {discrete_amounts.sum()}/{limit}"
    # )

    df_plot = pd.concat(
        [
            pd.DataFrame(
                {
                    "Word Classes": pos,
                    "Frequency (%)": percentages,
                    "Type": ["Percentages"] * len(percentages),
                }
            ),
            pd.DataFrame(
                {
                    "Word Classes": discrete_amounts.index,
                    "Frequency (%)": (discrete_amounts / limit) * 100,
                    "Type": ["Discrete Amounts (%)"] * len(discrete_amounts),
                }
            ),
        ],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    sns.barplot(
        x="Word Classes",
        y="Frequency (%)",
        hue="Type",
        data=df_plot,
        palette="muted",
        ax=ax,
    )
    ax.set_xlabel("Word Classes")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Comparison of Percentages and Discrete Amounts")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.close(fig)

    # plotting

    return discrete_amounts, (fig, ax)
