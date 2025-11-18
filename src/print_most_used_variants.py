"""
Script to extract and print the most used variants (traces) from the event log using pm4py.
"""

import pm4py
import pandas as pd
from pathlib import Path
from collections import Counter

# Configuration
LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")


def main():
    """Main function to extract and print most used variants."""
    
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
    
    print(f"Loaded {len(df):,} events across {df['case:concept:name'].nunique():,} cases")
    print("\nExtracting variants...")
    
    # Get variants using pm4py
    # This returns a dictionary mapping variant (tuple of activities) to count
    variants = pm4py.get_variants(df)
    
    # Convert to list of (variant, count) tuples and sort by count (descending)
    variant_counts = [(variant, count) for variant, count in variants.items()]
    variant_counts.sort(key=lambda x: x[1], reverse=True)
    
    total_cases = sum(count for _, count in variant_counts)
    unique_variants = len(variant_counts)
    
    print(f"\n{'='*80}")
    print("VARIANT ANALYSIS")
    print(f"{'='*80}")
    print(f"Total cases: {total_cases:,}")
    print(f"Unique variants: {unique_variants:,}")
    print(f"Variant coverage: {variant_counts[0][1]/total_cases*100:.2f}% for most common variant")
    print(f"\n{'='*80}")
    print("MOST USED VARIANTS")
    print(f"{'='*80}\n")
    
    # Print top N variants (default: top 20)
    top_n = 20
    print(f"Top {top_n} most used variants:\n")
    
    for rank, (variant, count) in enumerate(variant_counts[:top_n], 1):
        percentage = (count / total_cases) * 100
        variant_str = " -> ".join(str(activity) for activity in variant)
        print(f"{rank:3d}. Count: {count:6,} ({percentage:5.2f}%)")
        print(f"     Variant: {variant_str}")
        print()
    
    # Show summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if variant_counts:
        # Calculate coverage for top N variants
        for n in [1, 5, 10, 20, 50, 100]:
            top_n_count = sum(count for _, count in variant_counts[:n])
            coverage = (top_n_count / total_cases) * 100
            print(f"Top {n:3d} variants cover: {coverage:6.2f}% of cases ({top_n_count:,} cases)")
        
        # Show variants that appear only once
        unique_only = sum(1 for _, count in variant_counts if count == 1)
        print(f"\nVariants appearing only once: {unique_only:,} ({unique_only/unique_variants*100:.2f}% of unique variants)")
        
        # Show tail coverage (variants with count <= threshold)
        for threshold in [5, 10, 20]:
            tail_count = sum(1 for _, count in variant_counts if count <= threshold)
            print(f"Variants with count <= {threshold:2d}: {tail_count:,} ({tail_count/unique_variants*100:.2f}% of unique variants)")


if __name__ == "__main__":
    main()

