"""
Comprehensive visualization script for all attributes in the BPI Challenge 2017 event log.
Automatically distinguishes between case-level and event-level attributes and visualizes them appropriately.
Case-level attributes are visualized at the case level (one value per case), 
while event-level attributes are visualized at the event level.
"""

import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")

# Output directory for saving plots
OUTPUT_DIR = Path("Results/Attribute_Visualizations")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = r"Dataset\BPI Challenge 2017.xes"

# Threshold for classifying attributes as case-level
CASE_LEVEL_THRESHOLD = 80.0  # If >= 80% of cases have constant values, treat as case-level


def classify_attribute_level(df: pd.DataFrame, column: str, 
                             case_id_col: str = "case:concept:name") -> dict:
    """
    Classify an attribute as case-level, mostly case-level, or event-level.
    
    Returns a dictionary with:
    - 'level': 'case', 'mostly_case', or 'event'
    - 'percent_constant': Percentage of cases where the attribute is constant
    - 'is_case_level': Boolean indicating if it should be visualized at case level
    """
    if column == case_id_col:
        return {
            'level': 'event',  # Case ID is always event-level for visualization
            'percent_constant': 0.0,
            'is_case_level': False
        }
    
    # Group by case and check if all values in each case are the same
    case_groups = df.groupby(case_id_col)[column]
    
    def is_constant_in_case(series):
        """Check if all non-null values in a case are the same."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return True  # All NULL - consider as constant
        elif len(non_null) == 1:
            return True  # Only one non-null value - constant
        else:
            return non_null.nunique() <= 1
    
    is_constant_per_case = case_groups.apply(is_constant_in_case)
    total_cases = len(is_constant_per_case)
    num_cases_constant = is_constant_per_case.sum()
    percent_constant = (num_cases_constant / total_cases) * 100 if total_cases > 0 else 0.0
    
    # Classify based on threshold
    if percent_constant >= CASE_LEVEL_THRESHOLD:
        level = 'case'
        is_case_level = True
    elif percent_constant >= 50.0:
        level = 'mostly_case'
        is_case_level = True  # Visualize at case level
    else:
        level = 'event'
        is_case_level = False
    
    return {
        'level': level,
        'percent_constant': percent_constant,
        'is_case_level': is_case_level
    }


def get_case_level_values(df: pd.DataFrame, column: str, 
                          case_id_col: str = "case:concept:name") -> pd.Series:
    """
    Extract case-level values for an attribute.
    For each case, takes the first non-null value.
    """
    case_values = df.groupby(case_id_col)[column].apply(
        lambda x: x.dropna().iloc[0] if x.dropna().notna().any() else None
    )
    return case_values


def plot_case_level_categorical(df: pd.DataFrame, column: str, 
                                case_id_col: str = "case:concept:name",
                                top_n: int = 20, figsize: tuple = (12, 6)) -> None:
    """Plot bar chart for case-level categorical attributes."""
    case_values = get_case_level_values(df, column, case_id_col)
    case_values = case_values.dropna()
    
    if len(case_values) == 0:
        print(f"No data available for case-level {column}")
        return
    
    value_counts = case_values.value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar', color='steelblue')
    plt.title(f'Case-Level Distribution of {column}\n(Top {min(top_n, len(value_counts))}, {len(case_values):,} cases)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Number of Cases', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Use case_ prefix for case-level attributes
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'case_{filename}_bar.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_case_level_numerical(df: pd.DataFrame, column: str,
                              case_id_col: str = "case:concept:name",
                              figsize: tuple = (12, 6)) -> None:
    """Plot histogram and boxplot for case-level numerical attributes."""
    case_values = get_case_level_values(df, column, case_id_col)
    case_values = case_values.dropna()
    
    if len(case_values) == 0:
        print(f"No data available for case-level {column}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(case_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Case-Level Distribution of {column}\n({len(case_values):,} cases)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel(column, fontsize=10)
    axes[0].set_ylabel('Frequency (Number of Cases)', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Boxplot
    axes[1].boxplot(case_values, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].set_title(f'Case-Level Boxplot of {column}\n({len(case_values):,} cases)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel(column, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add statistics text
    stats_text = f'Mean: {case_values.mean():.2f}\nMedian: {case_values.median():.2f}\nStd: {case_values.std():.2f}'
    axes[1].text(1.1, case_values.median(), stats_text, 
                verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'case_{filename}_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_case_level_binary_categorical(df: pd.DataFrame, column: str,
                                       case_id_col: str = "case:concept:name",
                                       figsize: tuple = (10, 6)) -> None:
    """Plot pie chart and bar chart for case-level binary/limited categorical attributes."""
    case_values = get_case_level_values(df, column, case_id_col)
    case_values = case_values.dropna()
    
    if len(case_values) == 0:
        print(f"No data available for case-level {column}")
        return
    
    value_counts = case_values.value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    axes[0].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("husl", len(value_counts)))
    axes[0].set_title(f'Case-Level Distribution of {column}\n(Pie Chart, {len(case_values):,} cases)', 
                      fontsize=12, fontweight='bold')
    
    # Bar chart
    value_counts.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title(f'Case-Level Distribution of {column}\n(Bar Chart, {len(case_values):,} cases)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(column, fontsize=10)
    axes[1].set_ylabel('Number of Cases', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'case_{filename}_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_event_level_categorical(df: pd.DataFrame, column: str, 
                                 top_n: int = 20, figsize: tuple = (12, 6)) -> None:
    """Plot bar chart for event-level categorical attributes."""
    value_counts = df[column].value_counts().head(top_n)
    
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar', color='steelblue')
    plt.title(f'Event-Level Distribution of {column} (Top {min(top_n, len(value_counts))})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Number of Events', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'{filename}_bar.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_event_level_numerical(df: pd.DataFrame, column: str, 
                               figsize: tuple = (12, 6)) -> None:
    """Plot histogram and boxplot for event-level numerical attributes."""
    data = df[column].dropna()
    
    if len(data) == 0:
        print(f"No data available for {column}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Event-Level Distribution of {column}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(column, fontsize=10)
    axes[0].set_ylabel('Frequency (Number of Events)', fontsize=10)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Boxplot
    axes[1].boxplot(data, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[1].set_title(f'Event-Level Boxplot of {column}', fontsize=12, fontweight='bold')
    axes[1].set_ylabel(column, fontsize=10)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add statistics text
    stats_text = f'Mean: {data.mean():.2f}\nMedian: {data.median():.2f}\nStd: {data.std():.2f}'
    axes[1].text(1.1, data.median(), stats_text, 
                verticalalignment='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'{filename}_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_event_level_datetime(df: pd.DataFrame, column: str, 
                              figsize: tuple = (14, 6)) -> None:
    """Plot time series and distribution for event-level datetime attributes."""
    data = pd.to_datetime(df[column].dropna())
    
    if len(data) == 0:
        print(f"No data available for {column}")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Time series (events over time)
    daily_counts = data.dt.date.value_counts().sort_index()
    axes[0].plot(daily_counts.index, daily_counts.values, linewidth=1.5, color='steelblue')
    axes[0].set_title(f'Events Over Time - {column}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Date', fontsize=10)
    axes[0].set_ylabel('Number of Events', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Distribution by hour of day
    hourly_counts = data.dt.hour.value_counts().sort_index()
    axes[1].bar(hourly_counts.index, hourly_counts.values, color='steelblue', alpha=0.7)
    axes[1].set_title(f'Hourly Distribution - {column}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Hour of Day', fontsize=10)
    axes[1].set_ylabel('Number of Events', fontsize=10)
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'{filename}_timeseries.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_event_level_binary_categorical(df: pd.DataFrame, column: str, 
                                       figsize: tuple = (10, 6)) -> None:
    """Plot pie chart and bar chart for event-level binary/limited categorical attributes."""
    value_counts = df[column].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Pie chart
    axes[0].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
                startangle=90, colors=sns.color_palette("husl", len(value_counts)))
    axes[0].set_title(f'Event-Level Distribution of {column}\n(Pie Chart)', 
                      fontsize=12, fontweight='bold')
    
    # Bar chart
    value_counts.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title(f'Event-Level Distribution of {column}\n(Bar Chart)', 
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel(column, fontsize=10)
    axes[1].set_ylabel('Number of Events', fontsize=10)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    plt.savefig(OUTPUT_DIR / f'{filename}_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def plot_high_cardinality_attribute(df: pd.DataFrame, column: str,
                                    case_id_col: str = "case:concept:name",
                                    is_case_level: bool = False,
                                    figsize: tuple = (12, 6)) -> None:
    """Plot distribution statistics for high cardinality attributes (like IDs)."""
    if is_case_level:
        # Case-level statistics
        case_values = get_case_level_values(df, column, case_id_col)
        case_values = case_values.dropna()
        unique_count = case_values.nunique()
        total_count = len(case_values)
        level_type = "Case-Level"
        avg_text = f"Average Occurrences per Value: {total_count / unique_count:.2f}" if unique_count > 0 else "N/A"
    else:
        # Event-level statistics
        unique_count = df[column].nunique()
        total_count = df[column].notna().sum()
        level_type = "Event-Level"
        avg_text = f"Average Occurrences per Value: {total_count / unique_count:.2f}" if unique_count > 0 else "N/A"
    
    # Create a simple info plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    info_text = f"""
    {column} ({level_type})
    
    Total Count: {total_count:,}
    Unique Values: {unique_count:,}
    Duplicates: {total_count - unique_count:,}
    {avg_text}
    """
    
    ax.text(0.5, 0.5, info_text, ha='center', va='center', 
            fontsize=14, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_title(f'Statistics for {column} ({level_type})', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    filename = column.replace(":", "_")
    prefix = "case_" if is_case_level else ""
    plt.savefig(OUTPUT_DIR / f'{prefix}{filename}_stats.png', 
                dpi=300, bbox_inches='tight')
    plt.close()


def get_attribute_data_type(df: pd.DataFrame, column: str) -> str:
    """Determine the data type of an attribute for visualization purposes."""
    dtype = df[column].dtype
    
    # Check if datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'datetime'
    
    # Check if numerical
    if pd.api.types.is_numeric_dtype(dtype):
        return 'numerical'
    
    # Check if categorical
    if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
        # Check if binary or low cardinality
        unique_count = df[column].dropna().nunique()
        if unique_count <= 5:
            return 'binary_categorical'
        elif unique_count > 1000:  # High cardinality (likely IDs)
            return 'high_cardinality'
        else:
            return 'categorical'
    
    return 'categorical'  # Default


def main():
    """Main function to generate all visualizations."""
    print("Loading event log...")
    log = pm4py.read_xes(LOG_FILE_PATH)
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df)} events with {len(df.columns)} columns")
    print(f"Number of cases: {df['case:concept:name'].nunique():,}")
    print(f"Saving visualizations to: {OUTPUT_DIR}")
    print("\n" + "="*80)
    print("CLASSIFYING ATTRIBUTES")
    print("="*80)
    
    # Classify all attributes
    attribute_classifications = {}
    for column in df.columns:
        if column == 'case:concept:name':
            # Special handling for case ID
            attribute_classifications[column] = {
                'level': 'event',
                'percent_constant': 0.0,
                'is_case_level': False,
                'data_type': 'high_cardinality'
            }
            print(f"{column:35s} [event        ] "
                  f"{0.0:6.2f}% constant | "
                  f"EVENT-LEVEL | "
                  f"Type: high_cardinality")
        else:
            classification = classify_attribute_level(df, column)
            data_type = get_attribute_data_type(df, column)
            attribute_classifications[column] = {
                **classification,
                'data_type': data_type
            }
            print(f"{column:35s} [{classification['level']:12s}] "
                  f"{classification['percent_constant']:6.2f}% constant | "
                  f"{'CASE-LEVEL' if classification['is_case_level'] else 'EVENT-LEVEL'} | "
                  f"Type: {data_type}")
    
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Generate visualizations for each attribute
    for column, classification in attribute_classifications.items():
        is_case_level = classification['is_case_level']
        data_type = classification['data_type']
        
        print(f"\nProcessing {column}...")
        print(f"  Level: {classification['level']} ({'CASE-LEVEL' if is_case_level else 'EVENT-LEVEL'})")
        print(f"  Data Type: {data_type}")
        
        try:
            # Special handling for case:concept:name (case ID)
            if column == 'case:concept:name' and data_type == 'high_cardinality':
                plot_high_cardinality_attribute(df, column, is_case_level=False)
                print(f"  [OK] Created high cardinality visualization for {column}")
                continue
            
            # Case-level visualizations
            if is_case_level:
                if data_type == 'categorical':
                    plot_case_level_categorical(df, column)
                elif data_type == 'numerical':
                    plot_case_level_numerical(df, column)
                elif data_type == 'binary_categorical':
                    plot_case_level_binary_categorical(df, column)
                elif data_type == 'datetime':
                    # For datetime case-level, still show event-level time series
                    plot_event_level_datetime(df, column)
                elif data_type == 'high_cardinality':
                    plot_high_cardinality_attribute(df, column, is_case_level=True)
                else:
                    plot_case_level_categorical(df, column)  # Default
                print(f"  [OK] Created case-level visualization for {column}")
            
            # Event-level visualizations
            else:
                if data_type == 'categorical':
                    plot_event_level_categorical(df, column)
                elif data_type == 'numerical':
                    plot_event_level_numerical(df, column)
                elif data_type == 'binary_categorical':
                    plot_event_level_binary_categorical(df, column)
                elif data_type == 'datetime':
                    plot_event_level_datetime(df, column)
                elif data_type == 'high_cardinality':
                    plot_high_cardinality_attribute(df, column, is_case_level=False)
                else:
                    plot_event_level_categorical(df, column)  # Default
                print(f"  [OK] Created event-level visualization for {column}")
        
        except Exception as e:
            print(f"  [ERROR] Error creating visualization for {column}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("VISUALIZATION SUMMARY")
    print("="*80)
    num_files = len(list(OUTPUT_DIR.glob('*.png')))
    print(f"[OK] All visualizations saved to: {OUTPUT_DIR}")
    print(f"[OK] Total files created: {num_files}")
    print(f"[OK] Attributes processed: {len(attribute_classifications)}")


if __name__ == "__main__":
    main()

