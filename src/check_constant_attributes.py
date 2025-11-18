"""
Script to analyze constant attributes in the event log.
This script checks which attributes are constant across all cases (100% constant)
and which attributes vary across cases, showing the percentage of cases where each
attribute is constant.

Output format matches constant_attributes_analysis.txt
"""

import pm4py
import pandas as pd
from pathlib import Path

# Configuration
LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
OUTPUT_FILE_PATH = Path("Results/constant_attributes_analysis.txt")
CASE_ID_COL = "case:concept:name"


def is_constant_in_case(series: pd.Series) -> bool:
    """
    Check if all non-null values in a case are the same.
    
    Args:
        series: Series of values for a single case
        
    Returns:
        True if all non-null values are the same, False otherwise
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return True  # All NULL - consider as constant
    elif len(non_null) == 1:
        return True  # Only one non-null value - constant
    else:
        return non_null.nunique() <= 1


def analyze_attribute_constancy(df: pd.DataFrame, column: str) -> dict:
    """
    Analyze whether an attribute is constant across cases.
    
    Args:
        df: DataFrame containing the event log
        column: Name of the column/attribute to analyze
        
    Returns:
        Dictionary with:
        - 'is_constant': True if attribute is constant across ALL cases (100%)
        - 'percent_constant': Percentage of cases where the attribute is constant
        - 'total_cases': Total number of cases
        - 'constant_cases': Number of cases where attribute is constant
    """
    if column == CASE_ID_COL:
        # Case ID is always variable (by definition)
        return {
            'is_constant': False,
            'percent_constant': 0.0,
            'total_cases': df[CASE_ID_COL].nunique(),
            'constant_cases': 0
        }
    
    # Group by case and check if all values in each case are the same
    case_groups = df.groupby(CASE_ID_COL)[column]
    is_constant_per_case = case_groups.apply(is_constant_in_case)
    
    total_cases = len(is_constant_per_case)
    constant_cases = is_constant_per_case.sum()
    percent_constant = (constant_cases / total_cases * 100) if total_cases > 0 else 0.0
    
    return {
        'is_constant': (percent_constant == 100.0),
        'percent_constant': percent_constant,
        'total_cases': total_cases,
        'constant_cases': constant_cases
    }


def main():
    """Main function to analyze constant attributes."""
    
    # Check if log file exists
    log_path = Path(LOG_FILE_PATH)
    if not log_path.exists():
        # Try alternative path
        script_dir = Path(__file__).parent
        alt_log_path = script_dir.parent / LOG_FILE_PATH
        if alt_log_path.exists():
            log_path = alt_log_path
        else:
            raise FileNotFoundError(
                f"Could not find event log at: {LOG_FILE_PATH}\n"
                f"Tried: {log_path} and {alt_log_path}"
            )
    
    print(f"Loading event log from: {log_path}")
    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df):,} events with {len(df.columns)} attributes")
    print(f"Total cases: {df[CASE_ID_COL].nunique():,}")
    print("\nAnalyzing attribute constancy...")
    
    # Analyze all attributes
    constant_attributes = []
    variable_attributes = []
    attribute_results = {}
    
    for column in sorted(df.columns):
        print(f"  Analyzing: {column}")
        result = analyze_attribute_constancy(df, column)
        attribute_results[column] = result
        
        if result['is_constant']:
            constant_attributes.append(column)
        else:
            variable_attributes.append((column, result['percent_constant']))
    
    # Sort variable attributes by percent constant (descending)
    variable_attributes.sort(key=lambda x: x[1], reverse=True)
    
    # Create output directory if it doesn't exist
    OUTPUT_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Write results to file
    with open(OUTPUT_FILE_PATH, 'w') as f:
        f.write("CONSTANT ATTRIBUTES ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total attributes checked: {len(df.columns)}\n")
        f.write(f"Constant attributes: {len(constant_attributes)}\n")
        f.write(f"Variable attributes: {len(variable_attributes)}\n\n")
        
        f.write("CONSTANT ATTRIBUTES (stay the same across ALL cases):\n")
        f.write("-" * 80 + "\n")
        if constant_attributes:
            for attr in constant_attributes:
                f.write(f"  [CONSTANT] {attr}\n")
        else:
            f.write("  (none)\n")
        
        f.write("\nVARIABLE ATTRIBUTES:\n")
        f.write("-" * 80 + "\n")
        for attr, percent in variable_attributes:
            f.write(f"  [VARIABLE] {attr}: {percent:.2f}% of cases constant\n")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {OUTPUT_FILE_PATH}")
    print(f"\nSummary:")
    print(f"  Total attributes: {len(df.columns)}")
    print(f"  Constant attributes: {len(constant_attributes)}")
    print(f"  Variable attributes: {len(variable_attributes)}")
    
    if constant_attributes:
        print(f"\nConstant attributes:")
        for attr in constant_attributes:
            print(f"  - {attr}")
    
    # Show top 5 most constant variable attributes
    if variable_attributes:
        print(f"\nTop 5 most constant variable attributes:")
        for attr, percent in variable_attributes[:5]:
            print(f"  - {attr}: {percent:.2f}% constant")


if __name__ == "__main__":
    main()

