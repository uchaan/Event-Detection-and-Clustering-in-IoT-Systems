import pandas as pd
import matplotlib.pyplot as plt


def plot_mts_in_one_plot(df):
    variables = df.columns

    plt.figure(figsize=(12, 6))

    for variable in variables:
        plt.plot(df[variable], label=variable)

    plt.title("Multivariate Time Series Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_mts_in_one_column(info, df):
    variables = df.columns

    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 2 * len(variables)))

    plt.suptitle(info)

    for i, variable in enumerate(variables):
        axes[i].plot(df[variable])
        axes[i].set_title(variable)

    plt.tight_layout()
    plt.show()


def plot_anomaly(info: str, df: pd.DataFrame, labels: list, index: int = 1):
    variables = df.columns

    num_subplots = len(variables)

    fig, axes = plt.subplots(num_subplots, 1, figsize=(20, 2 * len(variables)))
    plt.suptitle(info)

    for i, variable in enumerate(variables):
        series = df[variable]
        ax = axes[i] if num_subplots > 1 else axes
        ax.scatter(range(len(series)), series, c=labels, cmap="coolwarm", marker=".")
        ax.set_title(variable)
        ax.set_xlabel("Time")

    plt.tight_layout()
    plt.show()
