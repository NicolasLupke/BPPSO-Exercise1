"""
Visualization: Resource distribution for the activity 'A_Create Application'.
Generates a bar chart of number of events per resource for that activity.
"""

from pathlib import Path

import pm4py
import pandas as pd
import matplotlib.pyplot as plt


LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
OUTPUT_DIR = Path("Results")
ACTIVITY_COLUMN = "concept:name"
RESOURCE_COLUMN = "org:resource"
LIFECYCLE_COLUMN = "lifecycle:transition"
TARGET_ACTIVITY = "A_Create Application"
# Set to True to restrict to lifecycle=complete when available
ONLY_COMPLETE = True


def main() -> None:
    print("Loading event log...")
    log_path = LOG_FILE_PATH
    if not log_path.exists():
        script_dir = Path(__file__).parent
        alt = script_dir.parent.parent / LOG_FILE_PATH
        if alt.exists():
            log_path = alt
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")

    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)

    if ACTIVITY_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {ACTIVITY_COLUMN}")
    if RESOURCE_COLUMN not in df.columns:
        raise ValueError(f"Missing column: {RESOURCE_COLUMN}")

    df_activity = df[df[ACTIVITY_COLUMN] == TARGET_ACTIVITY].copy()
    if ONLY_COMPLETE and LIFECYCLE_COLUMN in df.columns:
        df_activity = df_activity[df_activity[LIFECYCLE_COLUMN] == "complete"]

    total_events = len(df_activity)
    print(f"Filtered to '{TARGET_ACTIVITY}': {total_events:,} events")
    if total_events == 0:
        print("No events found for the specified activity; aborting.")
        return

    resource_counts = (
        df_activity[RESOURCE_COLUMN].dropna().value_counts().sort_values(ascending=False)
    )

    # Identify the label for "User_1" (robust to spacing/case)
    user1_detected_label = None
    for label in resource_counts.index.tolist():
        normalized = str(label).strip().lower().replace(" ", "_")
        if normalized == "user_1":
            user1_detected_label = label
            break
    # Fallback: try exact common casing
    if user1_detected_label is None and "User_1" in resource_counts.index:
        user1_detected_label = "User_1"

    user1_count = int(resource_counts.get(user1_detected_label, 0))
    others_count = int(total_events - user1_count)

    # Minimal pie chart: User_1 vs Others
    labels = [user1_detected_label or "User_1", "Others"]
    sizes = [user1_count, others_count]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(f"User_1 vs Others — {TARGET_ACTIVITY}\n({total_events:,} events)")
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pie_path = OUTPUT_DIR / "resource_distribution_A_Create_Application_pie.png"
    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved pie chart to: {pie_path}")

    # Save a detailed report of counts per resource
    counts_df = resource_counts.rename_axis("resource").reset_index(name="count")
    counts_csv = OUTPUT_DIR / "resource_distribution_A_Create_Application_counts.csv"
    counts_df.to_csv(counts_csv, index=False)
    print(f"[OK] Saved counts CSV to: {counts_csv}")


if __name__ == "__main__":
    main()


