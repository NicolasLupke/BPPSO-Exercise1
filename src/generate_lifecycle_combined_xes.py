"""
Script to generate a new XES file where concept:name and lifecycle:transition are combined.
The new activity name will be "{concept:name} - {lifecycle:transition}" or just "{concept:name}"
if lifecycle:transition is missing.
"""

import pm4py
import pandas as pd
from pathlib import Path

LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
OUTPUT_FILE_PATH = Path("Dataset/BPI Challenge 2017 - Lifecycle Combined.xes")
ACTIVITY_COLUMN = "concept:name"
LIFECYCLE_COLUMN = "lifecycle:transition"


def create_combined_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined activity column using activity and lifecycle.
    Replaces the original concept:name column with the combined version.
    """
    df = df.copy()
    
    # Check if lifecycle column exists
    if LIFECYCLE_COLUMN not in df.columns:
        print(f"Warning: '{LIFECYCLE_COLUMN}' column not found. Using only concept:name.")
        return df
    
    # Create combined activity name
    def combine_activity(row):
        activity = str(row[ACTIVITY_COLUMN]) if pd.notna(row[ACTIVITY_COLUMN]) else ""
        lifecycle = row[LIFECYCLE_COLUMN]
        
        if pd.notna(lifecycle):
            return f"{activity} - {lifecycle}"
        else:
            return activity
    
    # Replace concept:name with combined version
    df[ACTIVITY_COLUMN] = df.apply(combine_activity, axis=1)
    
    return df


def main():
    """Main function to generate the combined XES file."""
    print("Loading event log...")
    
    # Try relative path from project root first
    log_path = LOG_FILE_PATH
    if not log_path.exists():
        # If running from script directory, go up two levels
        script_dir = Path(__file__).parent
        log_path = script_dir.parent.parent / LOG_FILE_PATH
    
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")
    
    # Load the event log
    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df):,} events across {df['case:concept:name'].nunique():,} cases")
    print(f"Original unique activities: {df[ACTIVITY_COLUMN].nunique()}")
    
    # Create combined activity column
    print("\nCreating combined activity names...")
    df_combined = create_combined_activity_column(df)
    
    print(f"New unique activities: {df_combined[ACTIVITY_COLUMN].nunique()}")
    
    # Show some examples
    print("\nSample of combined activity names:")
    sample_activities = df_combined[ACTIVITY_COLUMN].unique()[:10]
    for activity in sample_activities:
        count = (df_combined[ACTIVITY_COLUMN] == activity).sum()
        print(f"  {activity}: {count:,} events")
    
    # Convert back to event log
    print("\nConverting to event log format...")
    log_combined = pm4py.convert_to_event_log(df_combined)
    
    # Determine output path
    output_path = OUTPUT_FILE_PATH
    if not output_path.is_absolute():
        script_dir = Path(__file__).parent
        output_path = script_dir.parent.parent / OUTPUT_FILE_PATH
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the new XES file
    print(f"\nWriting combined XES file to: {output_path}")
    pm4py.write_xes(log_combined, str(output_path))
    
    print(f"[OK] Successfully created combined XES file: {output_path}")
    print(f"[OK] Total events: {len(df_combined):,}")
    print(f"[OK] Total cases: {df_combined['case:concept:name'].nunique():,}")


if __name__ == "__main__":
    main()

