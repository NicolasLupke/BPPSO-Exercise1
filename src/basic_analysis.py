import pm4py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def format_duration(seconds, unit='days'):
    """Format duration in different units"""
    if unit == 'days':
        return seconds / 86400.0
    elif unit == 'hours':
        return seconds / 3600.0
    elif unit == 'minutes':
        return seconds / 60.0
    elif unit == 'seconds':
        return seconds
    else:
        return seconds


def main():
    # Load the event log
    print("Loading event log...")
    log = pm4py.read_xes("Dataset/BPI Challenge 2017.xes")
    
    # Calculate basic metrics
    num_cases = log['case:concept:name'].nunique()
    num_events = log["EventID"].nunique()
    num_offers = log['OfferID'].dropna().nunique()
    
    # Number of process variants
    variants = pm4py.get_variants(log)
    num_variants = len(variants)
    
    # Number of case and event labels
    case_labels = pm4py.get_trace_attributes(log)
    event_labels = pm4py.get_activity_labels(log)
    num_case_labels = len(case_labels)
    num_event_labels = len(event_labels)
    
    # Mean, median, min, max, and standard deviation of case length
    case_lengths = log.groupby('case:concept:name').size()
    mean_case_length = case_lengths.mean()
    median_case_length = case_lengths.median()
    min_case_length = case_lengths.min()
    max_case_length = case_lengths.max()
    std_case_length = case_lengths.std()
    
    # Mean, median, min, max, and standard deviation of case duration
    case_durations_seconds = pm4py.get_all_case_durations(log)
    mean_duration_seconds = np.mean(case_durations_seconds)
    median_duration_seconds = np.median(case_durations_seconds)
    min_duration_seconds = np.min(case_durations_seconds)
    max_duration_seconds = np.max(case_durations_seconds)
    std_duration_seconds = np.std(case_durations_seconds)
    
    # Convert to days
    mean_duration_days = format_duration(mean_duration_seconds, 'days')
    median_duration_days = format_duration(median_duration_seconds, 'days')
    min_duration_days = format_duration(min_duration_seconds, 'hours')
    max_duration_days = format_duration(max_duration_seconds, 'days')
    std_duration_days = format_duration(std_duration_seconds, 'days')
    

    # Optional metrics
    # 1. Number of unique activities
    num_unique_activities = len(event_labels)
    
    # 2. Case arrival average
    case_arrival_average_seconds = pm4py.stats.get_case_arrival_average(log)
    case_arrival_average_hours = format_duration(case_arrival_average_seconds, 'hours')
    
    
    # 4. Resources
    unique_resources = log['org:resource'].dropna().nunique()
    total_resource_events = log['org:resource'].notna().sum()
    
    # 5. Cycle time
    try:
        cycle_time = pm4py.stats.get_cycle_time(log)
    except:
        cycle_time = None
    
    # 6. Service time (average service time per activity)
    try:
        service_times = pm4py.stats.get_service_time(log)
        if service_times:
            avg_service_time = np.mean(list(service_times.values()))
        else:
            avg_service_time = None
    except:
        avg_service_time = None
    
    # 7. Activity counts (all activities sorted by count descending)
    activity_counts = log['concept:name'].value_counts().sort_values(ascending=False)
    activity_counts_df = pd.DataFrame({
        'Activity': activity_counts.index,
        'Count': [f"{count:,}" for count in activity_counts.values]
    })
    
    # Create statistics table
    stats_data = {
        'Metric': [
            'Number of cases',
            'Number of events',
            'Number of offers',
            'Number of process variants',
            'Number of case labels',
            'Number of event labels',
            'Number of unique activities',
            'Number of unique resources',
            'Mean case length',
            'Median case length',
            'Min case length',
            'Max case length',
            'Std dev case length',
            'Mean case duration (days)',
            'Median case duration (days)',
            'Min case duration (hours)',
            'Max case duration (days)',
            'Std dev case duration (days)',
            'Case arrival average (seconds)',
            'Case arrival average (hours)',
            'Cycle time (seconds)',
            'Average service time (seconds)'
        ],
        'Value': [
            f"{num_cases:,}",
            f"{num_events:,}",
            f"{num_offers:,}" if num_offers > 0 else "N/A",
            f"{num_variants:,}",
            num_case_labels,
            num_event_labels,
            num_unique_activities,
            unique_resources if unique_resources > 0 else "N/A",
            f"{mean_case_length:.2f}",
            f"{median_case_length:.2f}",
            f"{min_case_length:.0f}",
            f"{max_case_length:.0f}",
            f"{std_case_length:.2f}",
            f"{mean_duration_days:.2f}",
            f"{median_duration_days:.2f}",
            f"{min_duration_days:.2f}",
            f"{max_duration_days:.2f}",
            f"{std_duration_days:.2f}",
            f"{case_arrival_average_seconds:.2f}",
            f"{case_arrival_average_hours:.2f}",
            f"{cycle_time:.2f}" if cycle_time is not None else "N/A",
            f"{avg_service_time:.2f}" if avg_service_time is not None else "N/A"
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    
    # Display the table
    print("\n" + "="*80)
    print("BASIC EVENT LOG STATISTICS")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80)
    # Display activity counts table
    print("\n" + "="*80)
    print("ACTIVITY COUNTS (Sorted Descending)")
    print("="*80)
    print(activity_counts_df.to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    main()

