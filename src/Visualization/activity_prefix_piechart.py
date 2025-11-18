"""
Pie chart visualization script for the frequency of activity names by prefix.
Groups activities into:
- W_ prefix
- A_ prefix
- O_ prefix
- Other (activities with no prefix)
"""

import pm4py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory for saving plots
OUTPUT_DIR = Path("Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
ACTIVITY_COLUMN = "concept:name"


def extract_prefix_category(activity_name: str) -> str:
    """
    Extract the prefix category from an activity name.
    Returns 'W_', 'A_', 'O_', or 'Other'.
    """
    if pd.isna(activity_name):
        return 'Other'
    
    activity_name = str(activity_name)
    if activity_name.startswith('W_'):
        return 'W_'
    elif activity_name.startswith('A_'):
        return 'A_'
    elif activity_name.startswith('O_'):
        return 'O_'
    else:
        return 'Other'


def main():
    """Main function to create pie chart visualization."""
    print("Loading event log...")
    
    # Try relative path from project root first
    log_path = LOG_FILE_PATH
    if not log_path.exists():
        # If running from script directory, go up two levels
        script_dir = Path(__file__).parent
        log_path = script_dir.parent.parent / LOG_FILE_PATH
    
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")
    
    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df):,} events")
    print(f"Number of cases: {df['case:concept:name'].nunique():,}")
    print(f"Number of unique activities: {df[ACTIVITY_COLUMN].nunique():,}")
    
    # Extract prefix categories for all activities
    print("\nCategorizing activities by prefix...")
    df['prefix_category'] = df[ACTIVITY_COLUMN].apply(extract_prefix_category)
    
    # Count events by prefix category
    prefix_counts = df['prefix_category'].value_counts()
    
    print("\n" + "="*80)
    print("ACTIVITY PREFIX FREQUENCY")
    print("="*80)
    print(f"{'Category':<15} {'Count':<15} {'Percentage':<15}")
    print("-" * 45)
    
    total_events = len(df)
    for category in ['W_', 'A_', 'O_', 'Other']:
        count = prefix_counts.get(category, 0)
        percentage = (count / total_events * 100) if total_events > 0 else 0.0
        print(f"{category:<15} {count:>15,} {percentage:>14.2f}%")
    
    # Create pie chart
    print("\nCreating pie chart visualization...")
    
    # Get values and labels in the desired order (only include categories that exist)
    labels = []
    sizes = []
    
    for category in ['W_', 'A_', 'O_', 'Other']:
        if category in prefix_counts and prefix_counts[category] > 0:
            labels.append(category)
            sizes.append(prefix_counts[category])
    
    # Create minimal figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create simple pie chart
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = OUTPUT_DIR / 'activity_prefix_piechart.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Pie chart saved to: {output_path}")
    
    # Also show some additional statistics
    print("\n" + "="*80)
    print("ADDITIONAL STATISTICS")
    print("="*80)
    
    # Count unique activities by prefix
    unique_activities_by_prefix = df.groupby('prefix_category')[ACTIVITY_COLUMN].nunique()
    print(f"\nUnique activities by prefix category:")
    for category in ['W_', 'A_', 'O_', 'Other']:
        if category in unique_activities_by_prefix:
            count = unique_activities_by_prefix[category]
            print(f"  {category:<15} {count:>5} unique activities")
    
    plt.close()
    print("\n[OK] Visualization complete!")


if __name__ == "__main__":
    main()

