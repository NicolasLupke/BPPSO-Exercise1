import pm4py
import pandas as pd
import matplotlib.pyplot as plt


def plot_case_length_boxplot(log_path: str) -> None:
    """
    Load an XES event log and display a boxplot of case lengths
    (number of events per case).
    """
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    required_columns = {"case:concept:name"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"The event log is missing required columns: {missing}")

    case_lengths = df.groupby("case:concept:name").size().rename("length")
    lengths = case_lengths.dropna()
    if lengths.empty:
        raise ValueError("No case lengths available to plot.")

    plt.figure(figsize=(8, 6))
    plt.boxplot(lengths, vert=True, patch_artist=True)
    plt.ylabel("Case length (events)")
    plt.title("Case Length Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    LOG_FILE_PATH = r"Dataset\BPI Challenge 2017.xes"
    plot_case_length_boxplot(LOG_FILE_PATH)


