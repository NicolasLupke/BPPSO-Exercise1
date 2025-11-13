"""
Compute and visualise the case-arrival behaviour for applications that have been
created but have not yet progressed to `A_Pending`, `A_Cancelled`, or `A_Declined`.

The script:
1. Loads the BPI 2017 event log.
2. Filters to cases that contain `A_Create Application` and exclude any case
   that has already reached one of the specified terminal statuses.
3. Derives the first `A_Create Application` timestamp per remaining case.
4. Calculates the inter-arrival average via pm4py and plots the arrival rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Set

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import pm4py

# --------------------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------------------

LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
OUTPUT_DIR = Path("Results")
OUTPUT_PLOT = OUTPUT_DIR / "open_application_arrival_rate.png"

CASE_ID_KEY = "case:concept:name"
ACTIVITY_KEY = "concept:name"
TIMESTAMP_KEY = "time:timestamp"

CREATE_ACTIVITY = "A_Create Application"
EXCLUSION_ACTIVITIES: Set[str] = {
    "A_Pending",
    "A_Cancelled",
    "A_Declined",
}


# --------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------

def configure_matplotlib() -> None:
    """Apply a consistent plotting style with graceful fallbacks."""
    for style in ("seaborn-v0_8-darkgrid", "seaborn-darkgrid", "ggplot"):
        try:
            plt.style.use(style)
            return
        except OSError:
            continue


def load_event_dataframe(path: Path) -> pd.DataFrame:
    """Load the XES log and convert it to a pandas DataFrame."""
    print(f"Loading event log from: {path}")
    log = pm4py.read_xes(str(path))
    df = pm4py.convert_to_dataframe(log)
    df[TIMESTAMP_KEY] = pd.to_datetime(df[TIMESTAMP_KEY], utc=True)
    print(
        f"Loaded {len(df):,} events across "
        f"{df[CASE_ID_KEY].nunique():,} cases (unique activities: {df[ACTIVITY_KEY].nunique():,})."
    )
    return df


def select_open_cases(df: pd.DataFrame) -> pd.Index:
    """
    Return the case identifiers for applications that:
    - have an `A_Create Application` event, and
    - have NOT reached any exclusion activity.
    """
    create_cases: pd.Index = df.loc[
        df[ACTIVITY_KEY] == CREATE_ACTIVITY, CASE_ID_KEY
    ].unique()
    excluded_cases: pd.Index = df.loc[
        df[ACTIVITY_KEY].isin(EXCLUSION_ACTIVITIES), CASE_ID_KEY
    ].unique()
    open_cases = pd.Index(create_cases).difference(excluded_cases)
    print(
        f"Found {len(open_cases):,} open cases with '{CREATE_ACTIVITY}' "
        f"and without {sorted(EXCLUSION_ACTIVITIES)}."
    )
    return open_cases


def build_arrival_dataframe(df: pd.DataFrame, case_ids: Iterable[str]) -> pd.DataFrame:
    """
    Construct a DataFrame containing exactly one row per eligible case, holding
    the first `A_Create Application` timestamp. This works as a reduced log for
    pm4py statistics.
    """
    relevant = df[
        (df[CASE_ID_KEY].isin(case_ids)) & (df[ACTIVITY_KEY] == CREATE_ACTIVITY)
    ].copy()
    if relevant.empty:
        return pd.DataFrame(columns=[CASE_ID_KEY, TIMESTAMP_KEY, ACTIVITY_KEY])

    first_events = (
        relevant.sort_values(TIMESTAMP_KEY)
        .groupby(CASE_ID_KEY, as_index=False)
        .first()
    )
    arrival_df = first_events[[CASE_ID_KEY, TIMESTAMP_KEY]].copy()
    arrival_df[ACTIVITY_KEY] = CREATE_ACTIVITY
    arrival_df = arrival_df.sort_values(TIMESTAMP_KEY).reset_index(drop=True)

    min_ts = arrival_df[TIMESTAMP_KEY].min()
    max_ts = arrival_df[TIMESTAMP_KEY].max()
    print(
        f"Prepared reduced DataFrame with {len(arrival_df):,} arrival events "
        f"(from {min_ts} to {max_ts})."
    )
    return arrival_df


def aggregate_arrival_counts(
    arrival_df: pd.DataFrame, freq: str
) -> pd.Series:
    """Aggregate arrivals per time bucket based on the supplied pandas frequency."""
    if arrival_df.empty:
        return pd.Series(dtype="int64", name="new_cases")

    series = arrival_df.set_index(TIMESTAMP_KEY)[CASE_ID_KEY]
    counts = series.resample(freq).count().rename("new_cases")
    return counts


def plot_arrival_counts(counts: pd.Series, output_path: Path, freq: str) -> None:
    """Render and save a line plot for the arrival counts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        counts.index,
        counts.values,
        color="steelblue",
        linewidth=1.8,
        label="Open applications created",
    )

    window = max(2, len(counts) // 10) if len(counts) > 5 else None
    if window:
        trend = counts.rolling(window=window, min_periods=1).mean()
        ax.plot(
            trend.index,
            trend.values,
            color="darkorange",
            linewidth=2.2,
            label=f"{window}-period rolling mean",
        )

    ax.set_title(
        f"Arrivals of Open Applications ({CREATE_ACTIVITY}) - frequency: {freq}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of new open cases")
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
    print(f"Saved arrival rate plot to: {output_path}")


def compute_pm4py_arrival_average(arrival_df: pd.DataFrame) -> float:
    """Use pm4py to compute the average time between consecutive arrivals."""
    if arrival_df.empty:
        return 0.0

    avg_seconds = pm4py.get_case_arrival_average(
        arrival_df,
        activity_key=ACTIVITY_KEY,
        timestamp_key=TIMESTAMP_KEY,
        case_id_key=CASE_ID_KEY,
    )
    return float(avg_seconds)


# --------------------------------------------------------------------------------------
# CLI handling
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the arrival average of created applications that have not "
            "reached A_Pending, A_Cancelled, or A_Declined."
        )
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_FILE_PATH,
        help=f"Path to the event log (default: {LOG_FILE_PATH})",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="W",
        help="Resampling frequency for the plot (pandas offset alias, e.g. D/W/M).",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=OUTPUT_PLOT,
        help=f"Destination for the arrival plot (default: {OUTPUT_PLOT})",
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    df = load_event_dataframe(args.log_path)
    open_cases = select_open_cases(df)
    arrivals_df = build_arrival_dataframe(df, open_cases)

    if arrivals_df.empty:
        print("No eligible arrival events found; aborting.")
        return

    average_seconds = compute_pm4py_arrival_average(arrivals_df)
    if average_seconds <= 0:
        print("pm4py reported a non-positive average inter-arrival time.")
    else:
        cases_per_hour = 3600.0 / average_seconds
        cases_per_day = 86400.0 / average_seconds
        print(
            f"Average inter-arrival time: {average_seconds:.2f} seconds "
            f"({cases_per_hour:.2f} cases/hour, {cases_per_day:.2f} cases/day)."
        )

    arrival_counts = aggregate_arrival_counts(arrivals_df, args.frequency)
    if arrival_counts.empty:
        print("Arrival counts series is empty; skipping plot.")
    else:
        plot_arrival_counts(arrival_counts, args.output_plot, args.frequency)
        print("\nArrival count summary statistics:")
        print(arrival_counts.describe())


if __name__ == "__main__":
    main()


