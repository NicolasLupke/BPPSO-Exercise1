"""
Compute the percentage of traces (cases) that include the activity 'A_Concept'.
By default, considers only events with lifecycle:transition == 'complete' when available.
Outputs a brief report to Results and prints the percentage.
"""

from pathlib import Path

import pm4py
import pandas as pd


LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
OUTPUT_DIR = Path("Results")
CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
LIFECYCLE_COLUMN = "lifecycle:transition"
TARGET_ACTIVITY = "A_Concept"
ONLY_COMPLETE = True


def main() -> None:
    print("Loading event log...")
    log_path = LOG_FILE_PATH
    if not log_path.exists():
        script_dir = Path(__file__).parent
        alt = script_dir.parent / LOG_FILE_PATH
        if alt.exists():
            log_path = alt
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")

    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)

    if CASE_ID_COLUMN not in df.columns or ACTIVITY_COLUMN not in df.columns:
        raise ValueError(f"Missing required columns: {CASE_ID_COLUMN}, {ACTIVITY_COLUMN}")

    # Optionally restrict to lifecycle complete
    if ONLY_COMPLETE and LIFECYCLE_COLUMN in df.columns:
        df = df[df[LIFECYCLE_COLUMN] == "complete"].copy()

    total_cases = df[CASE_ID_COLUMN].nunique()
    print(f"Total cases: {total_cases:,}")

    # Cases containing A_Concept
    has_target = df[df[ACTIVITY_COLUMN] == TARGET_ACTIVITY]
    covered_cases = has_target[CASE_ID_COLUMN].nunique()
    coverage_pct = (covered_cases / total_cases * 100.0) if total_cases > 0 else 0.0

    print(f"Cases containing '{TARGET_ACTIVITY}': {covered_cases:,} "
          f"({coverage_pct:.2f}%)")

    # Save a simple report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "a_concept_trace_coverage.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("A_Concept Trace Coverage\n")
        f.write("========================\n")
        f.write(f"Only lifecycle 'complete': {ONLY_COMPLETE}\n")
        f.write(f"Total cases: {total_cases}\n")
        f.write(f"Cases containing '{TARGET_ACTIVITY}': {covered_cases}\n")
        f.write(f"Coverage: {coverage_pct:.2f}%\n")
    print(f"[OK] Saved report to: {out_path}")


if __name__ == "__main__":
    main()


