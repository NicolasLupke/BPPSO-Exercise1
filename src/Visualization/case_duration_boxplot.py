import pm4py
import pandas as pd
import matplotlib.pyplot as plt


def plot_case_duration_boxplot(log_path: str) -> None:
    """
    Load an XES event log and display a boxplot of case durations in days.
    """
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    required_columns = {"time:timestamp", "case:concept:name"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"The event log is missing required columns: {missing}")

    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])

    case_durations = (
        df.groupby("case:concept:name")["time:timestamp"]
        .agg(["min", "max"])
        .assign(duration_days=lambda x: (x["max"] - x["min"]).dt.total_seconds() / 86400.0)
    )

    durations = case_durations["duration_days"].dropna()
    if durations.empty:
        raise ValueError("No case durations available to plot.")

    plt.figure(figsize=(8, 6))
    plt.boxplot(durations, vert=True, patch_artist=True)
    plt.ylabel("Case duration (days)")
    plt.title("Case Duration Distribution")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    LOG_FILE_PATH = r"Dataset\BPI Challenge 2017.xes"
    plot_case_duration_boxplot(LOG_FILE_PATH)

