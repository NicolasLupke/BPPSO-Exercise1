"""
Analysis script to check if lifecycle:transition is mainly used for activities with W_ prefix.
"""

import pm4py
import pandas as pd
import numpy as np


def extract_prefix(activity_name: str) -> str:
    """
    Extract the prefix from an activity name.
    Returns 'W_', 'A_', 'O_', or 'other'.
    """
    if pd.isna(activity_name):
        return 'other'
    
    activity_name = str(activity_name)
    if activity_name.startswith('W_'):
        return 'W_'
    elif activity_name.startswith('A_'):
        return 'A_'
    elif activity_name.startswith('O_'):
        return 'O_'
    else:
        return 'other'


def main():
    """Main function to analyze lifecycle:transition usage by activity prefix."""
    # Load the event log
    print("Loading event log...")
    log = pm4py.read_xes("Dataset/BPI Challenge 2017.xes")
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df):,} events")
    print(f"Number of cases: {df['case:concept:name'].nunique():,}")
    print(f"Number of unique activities: {df['concept:name'].nunique():,}")
    
    # Check if lifecycle:transition column exists
    if 'lifecycle:transition' not in df.columns:
        print("\nERROR: 'lifecycle:transition' column not found in the event log!")
        return
    
    print("\n" + "="*80)
    print("LIFECYCLE:TRANSITION ANALYSIS")
    print("="*80)
    
    # Filter events with lifecycle:transition
    has_lifecycle = df['lifecycle:transition'].notna()
    events_with_lifecycle = df[has_lifecycle].copy()
    events_without_lifecycle = df[~has_lifecycle].copy()
    
    total_events = len(df)
    events_with_lt = len(events_with_lifecycle)
    events_without_lt = len(events_without_lifecycle)
    
    print(f"\nTotal events: {total_events:,}")
    print(f"Events with lifecycle:transition: {events_with_lt:,} ({events_with_lt/total_events*100:.2f}%)")
    print(f"Events without lifecycle:transition: {events_without_lt:,} ({events_without_lt/total_events*100:.2f}%)")
    
    # Extract prefixes for events with lifecycle:transition
    events_with_lifecycle['prefix'] = events_with_lifecycle['concept:name'].apply(extract_prefix)
    
    # Count by prefix
    prefix_counts = events_with_lifecycle['prefix'].value_counts()
    prefix_percentages = (prefix_counts / events_with_lt * 100).round(2)
    
    print("\n" + "="*80)
    print("LIFECYCLE:TRANSITION EVENTS BY ACTIVITY PREFIX")
    print("="*80)
    print(f"{'Prefix':<10} {'Count':<15} {'Percentage':<15}")
    print("-" * 40)
    for prefix in ['W_', 'A_', 'O_', 'other']:
        count = prefix_counts.get(prefix, 0)
        pct = prefix_percentages.get(prefix, 0.0)
        print(f"{prefix:<10} {count:>15,} {pct:>14.2f}%")
    
    # Get unique activities with lifecycle:transition by prefix
    activities_with_lt_by_prefix = {}
    for prefix in ['W_', 'A_', 'O_', 'other']:
        prefix_events = events_with_lifecycle[events_with_lifecycle['prefix'] == prefix]
        activities_with_lt_by_prefix[prefix] = sorted(prefix_events['concept:name'].unique().tolist())
    
    # Get all unique activities
    all_activities = sorted(df['concept:name'].unique().tolist())
    activities_with_lt = sorted(events_with_lifecycle['concept:name'].unique().tolist())
    activities_without_lt = sorted(set(all_activities) - set(activities_with_lt))
    
    # Get activities by prefix for all activities
    all_activities_by_prefix = {}
    for prefix in ['W_', 'A_', 'O_', 'other']:
        all_activities_by_prefix[prefix] = [act for act in all_activities if extract_prefix(act) == prefix]
    
    print("\n" + "="*80)
    print("ACTIVITIES WITH LIFECYCLE:TRANSITION BY PREFIX")
    print("="*80)
    
    for prefix in ['W_', 'A_', 'O_', 'other']:
        activities = activities_with_lt_by_prefix[prefix]
        if activities:
            print(f"\n{prefix} prefix ({len(activities)} activities):")
            for activity in activities:
                count = len(events_with_lifecycle[
                    (events_with_lifecycle['concept:name'] == activity) & 
                    (events_with_lifecycle['prefix'] == prefix)
                ])
                print(f"  - {activity} ({count:,} events)")
    
    print("\n" + "="*80)
    print("ACTIVITIES WITHOUT LIFECYCLE:TRANSITION BY PREFIX")
    print("="*80)
    
    activities_without_lt_by_prefix = {}
    for prefix in ['W_', 'A_', 'O_', 'other']:
        prefix_activities = [act for act in activities_without_lt if extract_prefix(act) == prefix]
        activities_without_lt_by_prefix[prefix] = prefix_activities
        if prefix_activities:
            print(f"\n{prefix} prefix ({len(prefix_activities)} activities):")
            for activity in prefix_activities:
                count = len(df[df['concept:name'] == activity])
                print(f"  - {activity} ({count:,} events)")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    w_activities_total = len(all_activities_by_prefix['W_'])
    w_activities_with_lt = len(activities_with_lt_by_prefix['W_'])
    w_activities_without_lt = len(activities_without_lt_by_prefix['W_'])
    
    a_activities_total = len(all_activities_by_prefix['A_'])
    a_activities_with_lt = len(activities_with_lt_by_prefix['A_'])
    a_activities_without_lt = len(activities_without_lt_by_prefix['A_'])
    
    o_activities_total = len(all_activities_by_prefix['O_'])
    o_activities_with_lt = len(activities_with_lt_by_prefix['O_'])
    o_activities_without_lt = len(activities_without_lt_by_prefix['O_'])
    
    other_activities_total = len(all_activities_by_prefix['other'])
    other_activities_with_lt = len(activities_with_lt_by_prefix['other'])
    other_activities_without_lt = len(activities_without_lt_by_prefix['other'])
    
    print(f"\nW_ prefix activities:")
    print(f"  Total: {w_activities_total}")
    print(f"  With lifecycle:transition: {w_activities_with_lt}")
    print(f"  Without lifecycle:transition: {w_activities_without_lt}")
    if w_activities_total > 0:
        print(f"  Percentage with lifecycle:transition: {w_activities_with_lt/w_activities_total*100:.2f}%")
    
    print(f"\nA_ prefix activities:")
    print(f"  Total: {a_activities_total}")
    print(f"  With lifecycle:transition: {a_activities_with_lt}")
    print(f"  Without lifecycle:transition: {a_activities_without_lt}")
    if a_activities_total > 0:
        print(f"  Percentage with lifecycle:transition: {a_activities_with_lt/a_activities_total*100:.2f}%")
    
    print(f"\nO_ prefix activities:")
    print(f"  Total: {o_activities_total}")
    print(f"  With lifecycle:transition: {o_activities_with_lt}")
    print(f"  Without lifecycle:transition: {o_activities_without_lt}")
    if o_activities_total > 0:
        print(f"  Percentage with lifecycle:transition: {o_activities_with_lt/o_activities_total*100:.2f}%")
    
    print(f"\nOther prefix activities:")
    print(f"  Total: {other_activities_total}")
    print(f"  With lifecycle:transition: {other_activities_with_lt}")
    print(f"  Without lifecycle:transition: {other_activities_without_lt}")
    if other_activities_total > 0:
        print(f"  Percentage with lifecycle:transition: {other_activities_with_lt/other_activities_total*100:.2f}%")
    
    # Answer the main question
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    w_percentage = prefix_percentages.get('W_', 0.0)
    if w_percentage >= 50.0:
        conclusion = f"YES - lifecycle:transition is mainly used for W_ prefixed activities ({w_percentage:.2f}% of all lifecycle:transition events)"
    else:
        conclusion = f"NO - lifecycle:transition is NOT mainly used for W_ prefixed activities ({w_percentage:.2f}% of all lifecycle:transition events)"
    
    print(f"\n{conclusion}")
    print(f"\nBreakdown:")
    print(f"  - W_ prefix: {prefix_counts.get('W_', 0):,} events ({prefix_percentages.get('W_', 0.0):.2f}%)")
    print(f"  - A_ prefix: {prefix_counts.get('A_', 0):,} events ({prefix_percentages.get('A_', 0.0):.2f}%)")
    print(f"  - O_ prefix: {prefix_counts.get('O_', 0):,} events ({prefix_percentages.get('O_', 0.0):.2f}%)")
    print(f"  - Other: {prefix_counts.get('other', 0):,} events ({prefix_percentages.get('other', 0.0):.2f}%)")
    
    # Additional insight: lifecycle transition values
    if events_with_lt > 0:
        lifecycle_values = events_with_lifecycle['lifecycle:transition'].value_counts()
        print("\n" + "="*80)
        print("LIFECYCLE:TRANSITION VALUES")
        print("="*80)
        print(f"{'Value':<20} {'Count':<15} {'Percentage':<15}")
        print("-" * 50)
        for value, count in lifecycle_values.items():
            pct = count / events_with_lt * 100
            print(f"{str(value):<20} {count:>15,} {pct:>14.2f}%")


if __name__ == "__main__":
    main()

