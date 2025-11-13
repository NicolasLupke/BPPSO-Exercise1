from pathlib import Path
from typing import Iterable, Sequence, Tuple

import pm4py
import pandas as pd


TARGET_ACTIVITIES: Tuple[str, ...] = ("A_Cancelled", "A_Denied", "A_Pending")
CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
LIFECYCLE_COLUMN = "lifecycle:transition"
DEFAULT_LOG_PATH = "C:\Users\nicol\Desktop\Repos\BPPSO - Assignment 1\Dataset\BPI Challenge 2017.xes"

def _validate_dataframe(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in (CASE_ID_COLUMN, ACTIVITY_COLUMN, LIFECYCLE_COLUMN)
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            "The event log must contain the columns "
            f"{', '.join((CASE_ID_COLUMN, ACTIVITY_COLUMN))}. "
            f"Missing: {', '.join(missing)}."
        )


def _filter_cases_by_activities(
    df: pd.DataFrame, target_activities: Iterable[str]
) -> pd.DataFrame:
    target_set = set(target_activities)
    if not target_set:
        raise ValueError("The list of target activities cannot be empty.")

    complete_events = df[df[LIFECYCLE_COLUMN] == "complete"]

    case_ids = (
        complete_events[complete_events[ACTIVITY_COLUMN].isin(target_set)][
            CASE_ID_COLUMN
        ]
        .dropna()
        .unique()
    )

    filtered = complete_events[complete_events[CASE_ID_COLUMN].isin(case_ids)]
    return filtered.copy()


def _load_log(
    log: object | None = None, log_path: str | Path | None = None
) -> pd.DataFrame:
    if isinstance(log, pd.DataFrame):
        return log.copy()

    if log is not None:
        return pm4py.convert_to_dataframe(log)

    path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    log_df = pm4py.read_xes(str(path))
    return pm4py.convert_to_dataframe(log_df)


def filter_log(
    log: object | None = None,
    activities: Sequence[str] | None = None,
    log_path: str | Path | None = None,
) -> object:
    """
    Filter a log so that only cases containing at least one target activity remain and
    return the result as a pm4py EventLog.

    Args:
        log: A pandas DataFrame or any pm4py log object (EventLog/EventStream/EventDF).
            If None, the log is loaded from `log_path`.
        activities: Activities to look for. Defaults to
            ("A_Cancelled", "A_Denied", "A_Pending").
        log_path: Path to an XES log file that should be read when `log` is None.
            Defaults to "Dataset/BPI Challenge 2017.xes".

    Returns:
        pm4py EventLog containing only the filtered cases, restricted to events whose
        lifecycle transition equals "complete".
    """
    activities = tuple(activities) if activities is not None else TARGET_ACTIVITIES

    df = _load_log(log, log_path)

    _validate_dataframe(df)
    filtered_df = _filter_cases_by_activities(df, activities)

    return pm4py.convert_to_event_log(filtered_df)
