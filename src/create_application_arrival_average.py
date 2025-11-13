"""
Compute the case arrival average for create-application events that have not yet
reached specific follow-up states.

The script filters the BPI Challenge 2017 event log to cases that:
  1. Contain the activity "A_Create Application"
  2. Have NOT reached any of the disqualifying states
     (default: A_Pending, A_Cancelled, A_Declined)

For those active cases, it calculates the average inter-arrival time between
their creation events using pm4py's statistics utilities, prints a concise
summary, and writes the details to a results file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd
import pm4py
from pm4py.statistics.traces.generic.pandas import case_statistics  # type: ignore

# Default configuration
LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
SUMMARY_OUTPUT_PATH = Path("Results/create_application_arrival_summary.txt")
DEFAULT_CREATE_ACTIVITY = "A_Create Application"
DEFAULT_EXCLUDED_STATES = ["A_Pending", "A_Cancelled", "A_Declined"]
CASE_ID_COL = "case:concept:name"
ACTIVITY_COL = "concept:name"
TIMESTAMP_COL = "time:timestamp"


def ensure_results_directory(path: Path) -> None:
    """Create the parent directory for the output path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def load_event_log(log_path: Path) -> pd.DataFrame:
    """Load the XES event log and return it as a pandas DataFrame."""
    print(f"Loading event log from: {log_path}")
    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    print(f"Loaded {len(df):,} events across {df[CASE_ID_COL].nunique():,} cases.")
    return df


def select_active_create_cases(
    df: pd.DataFrame,
    create_activity: str,
    excluded_states: Iterable[str],
) -> pd.DataFrame:
    """
    Filter the dataframe to cases that include the create activity and do not
    contain any of the excluded states.
    """
    excluded_states_set: Set[str] = set(excluded_states)
    print(f"Filtering cases: include '{create_activity}', exclude {sorted(excluded_states_set)}")

    case_activity_sets = df.groupby(CASE_ID_COL)[ACTIVITY_COL].agg(set)
    cases_with_create = {
        case_id for case_id, activities in case_activity_sets.items() if create_activity in activities
    }
    cases_with_excluded = {
        case_id
        for case_id, activities in case_activity_sets.items()
        if excluded_states_set.intersection(activities)
    }

    eligible_cases = cases_with_create.difference(cases_with_excluded)
    print(f"Eligible cases: {len(eligible_cases):,} "
          f"(cases with '{create_activity}': {len(cases_with_create):,}, "
          f"excluded cases: {len(cases_with_excluded):,})")

    if not eligible_cases:
        return df.iloc[0:0].copy()

    filtered_df = df[df[CASE_ID_COL].isin(eligible_cases)].copy()
    return filtered_df


def extract_create_events(df: pd.DataFrame, create_activity: str) -> pd.DataFrame:
    """Return a dataframe containing only the create activity events."""
    create_df = df[df[ACTIVITY_COL] == create_activity].copy()
    create_df = create_df.sort_values(TIMESTAMP_COL)
    print(f"Selected {len(create_df):,} '{create_activity}' events.")
    return create_df


def compute_arrival_series(df: pd.DataFrame) -> pd.Series:
    """Compute the sorted series of arrival timestamps via pm4py case statistics."""
    if df.empty:
        return pd.Series(dtype="datetime64[ns]", name="arrival_time")

    parameters: Dict = {
        case_statistics.Parameters.CASE_ID_KEY: CASE_ID_COL,
        case_statistics.Parameters.TIMESTAMP_KEY: TIMESTAMP_COL,
    }
    cases_description = case_statistics.get_cases_description(df, parameters=parameters)
    if not cases_description:
        return pd.Series(dtype="datetime64[ns]", name="arrival_time")

    cases_df = pd.DataFrame.from_dict(cases_description, orient="index")
    arrival_times = pd.to_datetime(cases_df["startTime"], unit="s", utc=True)
    arrival_series = pd.Series(arrival_times.values, index=cases_df.index, name="arrival_time")
    return arrival_series.sort_values()


def compute_case_arrival_average(df: pd.DataFrame) -> float:
    """Compute the average inter-arrival time (in seconds) using pm4py."""
    return pm4py.get_case_arrival_average(
        df,
        activity_key=ACTIVITY_COL,
        timestamp_key=TIMESTAMP_COL,
        case_id_key=CASE_ID_COL,
    )


def describe_arrival_times(arrival_series: pd.Series) -> pd.Series:
    """Return descriptive statistics for the arrival timestamp gaps."""
    if len(arrival_series) < 2:
        return pd.Series(dtype="float64")

    diffs = arrival_series.sort_values().diff().dropna()
    # Convert to hours for easier interpretation
    diffs_hours = diffs.dt.total_seconds() / 3600.0
    return diffs_hours.describe()


def format_seconds(seconds: float) -> str:
    """Return a human-friendly representation of seconds."""
    if seconds <= 0:
        return "N/A"
    minutes = seconds / 60.0
    hours = seconds / 3600.0
    days = seconds / 86400.0
    return f"{seconds:,.2f} s | {minutes:,.2f} min | {hours:,.2f} h | {days:,.2f} d"


def write_summary(
    output_path: Path,
    create_activity: str,
    excluded_states: Iterable[str],
    eligible_cases: int,
    create_events: int,
    arrival_average_seconds: float,
    arrival_series: pd.Series,
    resample_frequency: str | None,
) -> None:
    """Persist a textual summary of the computed statistics."""
    ensure_results_directory(output_path)

    lines = [
        "Create Application Arrival Analysis",
        "===================================",
        f"Create activity: {create_activity}",
        f"Excluded states: {', '.join(sorted(set(excluded_states)))}" if excluded_states else "Excluded states: none",
        f"Eligible cases: {eligible_cases:,}",
        f"Create events considered: {create_events:,}",
        "",
        "Average inter-arrival time between create events:",
        f"  {format_seconds(arrival_average_seconds)}",
    ]

    if arrival_average_seconds > 0:
        hourly_rate = 3600.0 / arrival_average_seconds
        daily_rate = 86400.0 / arrival_average_seconds
        lines.extend(
            [
                f"Equivalent throughput:",
                f"  {hourly_rate:,.2f} cases/hour",
                f"  {daily_rate:,.2f} cases/day",
            ]
        )

    stats = describe_arrival_times(arrival_series)
    if not stats.empty:
        lines.extend(
            [
                "",
                "Arrival gap statistics (hours):",
                stats.to_string(),
            ]
        )

    if resample_frequency and not arrival_series.empty:
        arrival_counts = (
            arrival_series.to_frame(name="arrival_time")
            .set_index("arrival_time")
            .resample(resample_frequency)
            .size()
        )
        lines.extend(
            [
                "",
                f"Arrival counts per {resample_frequency}:",
                arrival_counts.to_string(),
            ]
        )

    output_path.write_text("\n".join(lines))
    print(f"\nSummary written to: {output_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute case arrival average for create-application events that remain active."
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_FILE_PATH,
        help=f"Path to the XES log file (default: {LOG_FILE_PATH})",
    )
    parser.add_argument(
        "--create-activity",
        default=DEFAULT_CREATE_ACTIVITY,
        help=f"Activity to treat as the case creation event (default: {DEFAULT_CREATE_ACTIVITY})",
    )
    parser.add_argument(
        "--exclude-state",
        dest="exclude_states",
        action="append",
        default=None,
        help=("Repeat to list activities that disqualify a case (default: A_Pending, A_Cancelled, A_Declined). "
              "Use '--exclude-state A_Pending --exclude-state A_Cancelled' etc."),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=SUMMARY_OUTPUT_PATH,
        help=f"Where to write the textual summary (default: {SUMMARY_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--resample-frequency",
        default="W",
        help="Pandas offset alias for aggregating arrival counts (default: 'W' for weekly; use '' to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    exclude_states = args.exclude_states if args.exclude_states is not None else DEFAULT_EXCLUDED_STATES
    resample_frequency = args.resample_frequency if args.resample_frequency else None

    df = load_event_log(args.log_path)
    active_df = select_active_create_cases(df, args.create_activity, exclude_states)

    if active_df.empty:
        print("No cases satisfy the filtering criteria. Exiting.")
        ensure_results_directory(args.summary_path)
        args.summary_path.write_text("No cases satisfied the specified filtering criteria.")
        return

    create_events_df = extract_create_events(active_df, args.create_activity)
    if create_events_df.empty:
        print("No create events found after filtering. Exiting.")
        ensure_results_directory(args.summary_path)
        args.summary_path.write_text("No create events found after filtering.")
        return

    arrival_series = compute_arrival_series(create_events_df)
    arrival_average_seconds = compute_case_arrival_average(create_events_df)

    print("\n=== Create Application Arrival Summary ===")
    print(f"Eligible cases: {active_df[CASE_ID_COL].nunique():,}")
    print(f"Create events considered: {len(create_events_df):,}")
    print(f"Average inter-arrival time: {format_seconds(arrival_average_seconds)}")
    if arrival_average_seconds > 0:
        hourly_rate = 3600.0 / arrival_average_seconds
        daily_rate = 86400.0 / arrival_average_seconds
        print(f"Equivalent throughput: {hourly_rate:,.2f} cases/hour | {daily_rate:,.2f} cases/day")

    write_summary(
        args.summary_path,
        args.create_activity,
        exclude_states,
        active_df[CASE_ID_COL].nunique(),
        len(create_events_df),
        arrival_average_seconds,
        arrival_series,
        resample_frequency,
    )


if __name__ == "__main__":
    main()


