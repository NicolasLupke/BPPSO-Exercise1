"""
Lifecycle-aware variants analysis.

Builds variants using combined activity labels: "{concept:name} - {lifecycle:transition}"
(falls back to just concept:name if lifecycle is missing), then prints the most
frequent variants and basic coverage statistics.
"""

from pathlib import Path
from collections import Counter
from typing import List, Tuple

import pm4py
import pandas as pd


LOG_FILE_PATH = Path("Dataset/BPI Challenge 2017.xes")
CASE_ID_COLUMN = "case:concept:name"
ACTIVITY_COLUMN = "concept:name"
LIFECYCLE_COLUMN = "lifecycle:transition"
TIMESTAMP_COLUMN = "time:timestamp"


def create_combined_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    """Create a combined activity column using activity and lifecycle."""
    df = df.copy()
    if LIFECYCLE_COLUMN not in df.columns:
        df[LIFECYCLE_COLUMN] = pd.NA
    df["combined_activity"] = df.apply(
        lambda row: (
            f"{row[ACTIVITY_COLUMN]} - {row[LIFECYCLE_COLUMN]}"
            if pd.notna(row[LIFECYCLE_COLUMN])
            else str(row[ACTIVITY_COLUMN])
        ),
        axis=1,
    )
    return df


def compute_variants(df: pd.DataFrame) -> List[Tuple[Tuple[str, ...], int]]:
    """
    Compute variants as ordered sequences of combined activities per case.
    Returns a sorted list of (variant_tuple, count) in descending count order.
    """
    # Ensure timestamp exists for ordering; if missing, rely on index order
    sort_cols = [CASE_ID_COLUMN]
    if TIMESTAMP_COLUMN in df.columns:
        sort_cols.append(TIMESTAMP_COLUMN)
    df_sorted = df.sort_values(sort_cols)

    traces = (
        df_sorted.groupby(CASE_ID_COLUMN)["combined_activity"]
        .apply(tuple)
        .tolist()
    )
    counts = Counter(traces)
    variant_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return variant_counts


def main() -> None:
    print("Loading event log...")
    log_path = LOG_FILE_PATH
    if not log_path.exists():
        script_dir = Path(__file__).parent
        alt_log_path = script_dir.parent / LOG_FILE_PATH
        if alt_log_path.exists():
            log_path = alt_log_path
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")

    log = pm4py.read_xes(str(log_path))
    df = pm4py.convert_to_dataframe(log)

    print(f"Loaded {len(df):,} events across {df[CASE_ID_COLUMN].nunique():,} cases")
    print(f"Columns: {', '.join(df.columns)}")

    print("\nCreating lifecycle-aware activity labels...")
    df = create_combined_activity_column(df)

    print("Computing variants...")
    variant_counts = compute_variants(df)
    total_cases = len(variant_counts) and sum(count for _, count in variant_counts) or 0
    unique_variants = len(variant_counts)

    print("\n" + "=" * 80)
    print("LIFECYCLE-AWARE VARIANTS")
    print("=" * 80)
    print(f"Total cases: {total_cases:,}")
    print(f"Unique variants: {unique_variants:,}")
    if total_cases > 0 and unique_variants > 0:
        print(
            f"Coverage of top variant: {variant_counts[0][1]/total_cases*100:.2f}%"
        )

    top_n = 20
    print("\n" + "=" * 80)
    print(f"TOP {top_n} VARIANTS")
    print("=" * 80)
    for rank, (variant, count) in enumerate(variant_counts[:top_n], 1):
        pct = (count / total_cases * 100) if total_cases > 0 else 0.0
        variant_str = " -> ".join(variant)
        print(f"{rank:3d}. Count: {count:6,} ({pct:5.2f}%)")
        print(f"     {variant_str}")

    # Save results
    output_dir = Path("Results")
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "lifecycle_variants_summary.txt"
    csv_path = output_dir / "lifecycle_variants.csv"

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("LIFECYCLE-AWARE VARIANTS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total cases: {total_cases}\n")
        f.write(f"Unique variants: {unique_variants}\n")
        if total_cases > 0 and unique_variants > 0:
            f.write(
                f"Coverage of top variant: {variant_counts[0][1]/total_cases*100:.2f}%\n"
            )
        f.write("\nTop variants:\n")
        for rank, (variant, count) in enumerate(variant_counts[:top_n], 1):
            pct = (count / total_cases * 100) if total_cases > 0 else 0.0
            variant_str = " -> ".join(variant)
            f.write(f"{rank:3d}. {count} ({pct:5.2f}%): {variant_str}\n")

    # Flatten to CSV
    rows = []
    for variant, count in variant_counts:
        rows.append(
            {
                "count": count,
                "percentage": (count / total_cases * 100) if total_cases > 0 else 0.0,
                "variant_length": len(variant),
                "variant": " -> ".join(variant),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print(f"[OK] Saved summary to: {txt_path}")
    print(f"[OK] Saved full variants to: {csv_path}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()


