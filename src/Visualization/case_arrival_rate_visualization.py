"""
Visualize the arrival rate of new cases in the BPI Challenge 2017 event log.

The script extracts the first event timestamp for every case, aggregates the
number of new cases per configurable time bucket, and saves a line chart showing
the arrival rate together with an optional rolling average trend.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pm4py
from pm4py.statistics.traces.generic.pandas import case_statistics  # type: ignore


# Default paths / constants
LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
RESULTS_DIR = Path("Results")
OUTPUT_PATH = RESULTS_DIR / "arrival_rate_over_time.png"


def configure_plot_style() -> None:
    """Configure matplotlib style with sensible fallbacks."""
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
        try:
            plt.style.use(style)
            break
        except OSError:
            continue


def load_event_log(log_path: Path) -> pd.DataFrame:
    """Load the XES event log and return it as a pandas DataFrame."""
    print(f"Loading event log from: {log_path}")
    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)
    if "time:timestamp" not in df.columns:
        raise KeyError("Column 'time:timestamp' not found in the event log.")
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    print(f"Loaded {len(df):,} events across {df['case:concept:name'].nunique():,} cases.")
    return df


def compute_case_arrival_times(
    df: pd.DataFrame,
    case_id_col: str = "case:concept:name",
    timestamp_col: str = "time:timestamp",
) -> pd.Series:
    """
    Compute case arrival timestamps using pm4py's case statistics utility.

    Returns a pandas Series indexed by case identifier with arrival timestamps.
    """
    print("Computing case arrival timestamps via pm4py...")
    parameters: Dict = {
        case_statistics.Parameters.CASE_ID_KEY: case_id_col,
        case_statistics.Parameters.TIMESTAMP_KEY: timestamp_col,
    }
    cases_info = case_statistics.get_cases_description(df, parameters=parameters)
    if not cases_info:
        return pd.Series(dtype="datetime64[ns]", name="arrival_time")

    cases_df = pd.DataFrame.from_dict(cases_info, orient="index")
    # pm4py returns epoch seconds; convert to timezone-aware datetime
    arrival_times = pd.to_datetime(cases_df["startTime"], unit="s", utc=True)
    arrival_series = pd.Series(arrival_times.values, index=cases_df.index, name="arrival_time")
    arrival_series = arrival_series.sort_values()
    return arrival_series


def aggregate_arrival_counts(
    arrivals: pd.Series,
    frequency: str,
) -> pd.Series:
    """
    Aggregate arrival counts per time bucket using the provided pandas frequency code.

    Parameters
    ----------
    arrivals : Series
        Series of arrival timestamps indexed by case id.
    frequency : str
        Pandas offset alias (e.g., 'D' for daily, 'W' for weekly, 'M' for monthly).
    """
    print(f"Aggregating arrivals with frequency '{frequency}'...")
    arrival_counts = (
        arrivals.to_frame()
        .set_index("arrival_time")
        .resample(frequency)
        .size()
        .rename("new_cases")
    )
    return arrival_counts


def plot_arrival_rate(
    arrival_counts: pd.Series,
    rolling_window: int | None,
    output_path: Path,
) -> None:
    """Create and save the arrival rate visualization."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        arrival_counts.index,
        arrival_counts.values,
        label="New cases",
        color="steelblue",
        linewidth=1.8,
    )

    if rolling_window and rolling_window > 1:
        trend = arrival_counts.rolling(window=rolling_window, min_periods=1).mean()
        ax.plot(
            trend.index,
            trend.values,
            label=f"{rolling_window}-period rolling average",
            color="darkorange",
            linewidth=2.2,
        )

    ax.set_title("Arrival Rate of New Cases", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of new cases")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved arrival rate visualization to: {output_path}")


def parse_frequency(value: str) -> str:
    """Validate human-friendly frequency choices and map them to pandas aliases."""
    value_lower = value.lower()
    mapping = {
        "daily": "D",
        "weekly": "W",
        "monthly": "M",
        "quarterly": "Q",
    }
    if value_lower in mapping:
        return mapping[value_lower]

    # Allow raw pandas offset aliases for advanced users.
    try:
        pd.Timedelta("1" + value.upper())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid frequency '{value}'. Use one of {list(mapping.keys())} or a pandas offset alias."
        ) from exc
    return value.upper()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the arrival rate of new cases in the event log.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_FILE_PATH,
        help=f"Path to the XES log file (default: {LOG_FILE_PATH})",
    )
    parser.add_argument(
        "--frequency",
        type=parse_frequency,
        default="W",
        help="Aggregation frequency for arrivals (daily, weekly, monthly, quarterly, or pandas alias). Default: weekly.",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=4,
        help="Rolling window (in number of periods) for the trend line. Use 0 to disable. Default: 4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path for the plot (default: {OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    configure_plot_style()
    args = parse_arguments()

    df = load_event_log(args.log_path)
    arrivals = compute_case_arrival_times(df)
    arrival_counts = aggregate_arrival_counts(arrivals, args.frequency)

    if arrival_counts.empty:
        print("No arrival data found. Exiting without creating a plot.")
        return

    rolling_window = args.rolling_window if args.rolling_window and args.rolling_window > 1 else None
    plot_arrival_rate(arrival_counts, rolling_window, args.output)

    average_interarrival = pm4py.get_case_arrival_average(
        df,
        timestamp_key="time:timestamp",
        case_id_key="case:concept:name",
    )
    print(f"Average inter-arrival time (seconds) via pm4py: {average_interarrival:.2f}")
    if average_interarrival > 0:
        hourly_rate = 3600.0 / average_interarrival
        daily_rate = 86400.0 / average_interarrival
        print(f"Equivalent arrival rate: {hourly_rate:.2f} cases/hour | {daily_rate:.2f} cases/day")

    print("\nSummary statistics:")
    print(arrival_counts.describe())


if __name__ == "__main__":
    main()


