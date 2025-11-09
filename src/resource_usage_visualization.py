import pm4py
import pandas as pd
import matplotlib.pyplot as plt


def plot_resource_usage(log_path: str, top_n: int | None = None) -> None:
    """
    Load an XES event log and plot resource usage as a bar chart.

    Args:
        log_path: Path to the XES file.
        top_n: If provided, limit the chart to the top N most active resources.
    """
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    resource_column = "org:resource"
    if resource_column not in df.columns:
        raise ValueError(f"The event log must contain the '{resource_column}' column.")

    resource_counts = (
        df[resource_column]
        .dropna()
        .value_counts()
        .sort_values(ascending=False)
    )

    if resource_counts.empty:
        raise ValueError("No resource usage information available to plot.")

    if top_n is not None:
        resource_counts = resource_counts.head(top_n)

    plt.figure(figsize=(10, 6))
    resource_counts.plot(kind="bar", color="#4C72B0")
    plt.xlabel("Resource")
    plt.ylabel("Number of events")
    plt.title("Resource Usage")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    LOG_FILE_PATH = r"Dataset\BPI Challenge 2017.xes"
    plot_resource_usage(LOG_FILE_PATH, top_n=20)

