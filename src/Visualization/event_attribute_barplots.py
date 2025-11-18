import pm4py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory for saving plots
OUTPUT_DIR = Path("Results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE_PATH = r"Dataset\BPI Challenge 2017.xes"


def plot_attribute_barplot(df: pd.DataFrame, column: str, output_filename: str, 
                           title: str | None = None, figsize: tuple = (10, 6)) -> None:
    """
    Create a barplot for a specific attribute column.
    
    Args:
        df: DataFrame containing the event log data
        column: Name of the column to plot
        output_filename: Filename for saving the plot (without path)
        title: Optional title for the plot. If None, uses column name
        figsize: Figure size tuple (width, height)
    """
    if column not in df.columns:
        print(f"Warning: Column '{column}' not found in event log. Skipping...")
        return
    
    # Get value counts, dropping NaN values
    value_counts = df[column].dropna().value_counts().sort_values(ascending=False)
    
    if value_counts.empty:
        print(f"Warning: No data available for '{column}'. Skipping...")
        return
    
    # Create the plot
    plt.figure(figsize=figsize)
    value_counts.plot(kind="bar", color="#4C72B0")
    
    # Set labels and title
    plot_title = title if title is not None else f"{column} Distribution"
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Number of events", fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=45, ha="right")
    
    # Add grid
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = OUTPUT_DIR / output_filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved barplot for '{column}' to {output_path}")


def main():
    """Main function to load event log and generate all barplots."""
    print("Loading event log...")
    log = pm4py.read_xes(LOG_FILE_PATH)
    df = pm4py.convert_to_dataframe(log)
    
    print(f"Loaded {len(df):,} events across {df['case:concept:name'].nunique():,} cases")
    print(f"Generating barplots for specified attributes...\n")
    
    # Define the attributes to visualize with their output filenames
    attributes = [
        ("Action", "action_barplot.png", "Action Distribution"),
        ("Selected", "selected_barplot.png", "Selected Distribution"),
        ("case:ApplicationType", "applicationType_barplot.png", "Application Type Distribution"),
        ("case:LoanGoal", "loanGoal_barplot.png", "Loan Goal Distribution"),
        ("EventOrigin", "eventOrigin_barplot.png", "Event Origin Distribution"),
        ("Accepted", "accepted_barplot.png", "Accepted Distribution"),
    ]
    
    # Generate barplots for each attribute
    for column, filename, title in attributes:
        plot_attribute_barplot(df, column, filename, title)
    
    print(f"\nAll barplots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

