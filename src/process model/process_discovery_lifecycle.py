"""
Process discovery script using combined concept:name and lifecycle:transition as activity names.
This script performs process discovery using Alpha Miner, Inductive Miner, and Heuristic Miner
with activity names that combine concept:name and lifecycle:transition.
"""

import pm4py
import pandas as pd
from pathlib import Path


def create_combined_activity_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined activity column that combines concept:name and lifecycle:transition.
    
    Args:
        df: DataFrame with 'concept:name' and 'lifecycle:transition' columns
        
    Returns:
        DataFrame with new 'combined_activity' column
    """
    df = df.copy()
    
    # Combine concept:name and lifecycle:transition
    # If lifecycle:transition is NaN, use just concept:name
    df['combined_activity'] = df.apply(
        lambda row: (
            f"{row['concept:name']} - {row['lifecycle:transition']}"
            if pd.notna(row['lifecycle:transition'])
            else row['concept:name']
        ),
        axis=1
    )
    
    return df


def print_model_statistics(name: str, net, initial_marking, final_marking):
    """
    Print statistics about a discovered Petri net model.
    
    Args:
        name: Name of the miner/model
        net: Petri net
        initial_marking: Initial marking
        final_marking: Final marking
    """
    print(f"\n{'='*80}")
    print(f"{name} - Model Statistics")
    print(f"{'='*80}")
    
    # Count places and transitions
    places = list(net.places)
    transitions = list(net.transitions)
    arcs = list(net.arcs)
    
    # Count visible transitions (non-silent)
    visible_transitions = [t for t in transitions if t.label is not None]
    
    print(f"Places: {len(places)}")
    print(f"Transitions: {len(transitions)}")
    print(f"  - Visible transitions: {len(visible_transitions)}")
    print(f"  - Silent transitions: {len(transitions) - len(visible_transitions)}")
    print(f"Arcs: {len(arcs)}")
    print(f"Initial marking size: {sum(initial_marking.values())}")
    print(f"Final marking size: {sum(final_marking.values())}")


def main():
    """Main function to perform lifecycle-aware process discovery."""
    
    # Load event log
    print("Loading event log...")
    # Try relative path from project root first, then try relative from script location
    log_path = Path("Dataset/BPI Challenge 2017.xes")
    if not log_path.exists():
        # If running from script directory, go up two levels
        script_dir = Path(__file__).parent
        log_path = script_dir.parent.parent / "Dataset" / "BPI Challenge 2017.xes"
    
    if not log_path.exists():
        raise FileNotFoundError(f"Could not find event log at: {log_path}")
    
    log = pm4py.read_xes(str(log_path))
    
    # Convert to DataFrame to add combined activity column
    print("Creating combined activity column...")
    df = pm4py.convert_to_dataframe(log)
    
    # Check if lifecycle:transition column exists
    if 'lifecycle:transition' not in df.columns:
        print("WARNING: 'lifecycle:transition' column not found. Using only concept:name.")
        df['lifecycle:transition'] = pd.NA
    
    # Create combined activity column
    df = create_combined_activity_column(df)
    
    # Show some statistics
    total_events = len(df)
    events_with_lifecycle = df['lifecycle:transition'].notna().sum()
    unique_activities_original = df['concept:name'].nunique()
    unique_activities_combined = df['combined_activity'].nunique()
    
    print(f"\nEvent log statistics:")
    print(f"  Total events: {total_events:,}")
    print(f"  Events with lifecycle:transition: {events_with_lifecycle:,} ({events_with_lifecycle/total_events*100:.2f}%)")
    print(f"  Unique activities (original): {unique_activities_original}")
    print(f"  Unique activities (combined): {unique_activities_combined}")
    
    # Convert back to event log format
    log_combined = pm4py.convert_to_event_log(df)
    
    # Create output directory if it doesn't exist
    # Try relative path from project root first
    output_dir = Path("Results/Models")
    if not output_dir.exists():
        # If running from script directory, go up two levels
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent.parent / "Results" / "Models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # Alpha Miner
    # ============================================================================
    print(f"\n{'='*80}")
    print("Running Alpha Miner...")
    print(f"{'='*80}")
    
    try:
        alpha_net, alpha_im, alpha_fm = pm4py.discover_petri_net_alpha(
            log_combined,
            activity_key="combined_activity"
        )
        
        print_model_statistics("Alpha Miner", alpha_net, alpha_im, alpha_fm)
        
        # Convert to BPMN
        alpha_bpmn = pm4py.convert_to_bpmn(alpha_net, alpha_im, alpha_fm)
        
        # Save visualizations
        alpha_petri_path = output_dir / "Alpha Miner Lifecycle PetriNet.jpg"
        alpha_bpmn_path = output_dir / "Alpha Miner Lifecycle BPMN.jpg"
        
        pm4py.save_vis_petri_net(alpha_net, alpha_im, alpha_fm, str(alpha_petri_path))
        pm4py.save_vis_bpmn(alpha_bpmn, str(alpha_bpmn_path))
        
        print(f"\nVisualizations saved:")
        print(f"  Petri Net: {alpha_petri_path}")
        print(f"  BPMN: {alpha_bpmn_path}")
        
    except Exception as e:
        print(f"Error running Alpha Miner: {e}")
        import traceback
        traceback.print_exc()
        alpha_net, alpha_im, alpha_fm = None, None, None
        alpha_bpmn = None
    
    # ============================================================================
    # Inductive Miner
    # ============================================================================
    print(f"\n{'='*80}")
    print("Running Inductive Miner...")
    print(f"{'='*80}")
    
    try:
        inductive_net, inductive_im, inductive_fm = pm4py.discover_petri_net_inductive(
            log_combined,
            activity_key="combined_activity"
        )
        
        print_model_statistics("Inductive Miner", inductive_net, inductive_im, inductive_fm)
        
        # Convert to BPMN
        inductive_bpmn = pm4py.convert_to_bpmn(inductive_net, inductive_im, inductive_fm)
        
        # Save visualizations
        inductive_petri_path = output_dir / "Inductive Miner Lifecycle PetriNet.jpg"
        inductive_bpmn_path = output_dir / "Inductive Miner Lifecycle BPMN.jpg"
        
        pm4py.save_vis_petri_net(inductive_net, inductive_im, inductive_fm, str(inductive_petri_path))
        pm4py.save_vis_bpmn(inductive_bpmn, str(inductive_bpmn_path))
        
        print(f"\nVisualizations saved:")
        print(f"  Petri Net: {inductive_petri_path}")
        print(f"  BPMN: {inductive_bpmn_path}")
        
    except Exception as e:
        print(f"Error running Inductive Miner: {e}")
        import traceback
        traceback.print_exc()
        inductive_net, inductive_im, inductive_fm = None, None, None
        inductive_bpmn = None
    
    # ============================================================================
    # Heuristic Miner
    # ============================================================================
    print(f"\n{'='*80}")
    print("Running Heuristic Miner...")
    print(f"{'='*80}")
    
    try:
        heuristic_net, heuristic_im, heuristic_fm = pm4py.discover_petri_net_heuristics(
            log_combined,
            activity_key="combined_activity",
            dependency_threshold=0.5,
            and_threshold=0.65,
            loop_two_threshold=0.5
        )
        
        print_model_statistics("Heuristic Miner", heuristic_net, heuristic_im, heuristic_fm)
        
        # Convert to BPMN
        heuristic_bpmn = pm4py.convert_to_bpmn(heuristic_net, heuristic_im, heuristic_fm)
        
        # Save visualizations
        heuristic_petri_path = output_dir / "Heuristic Miner Lifecycle PetriNet.jpg"
        heuristic_bpmn_path = output_dir / "Heuristic Miner Lifecycle BPMN.jpg"
        
        pm4py.save_vis_petri_net(heuristic_net, heuristic_im, heuristic_fm, str(heuristic_petri_path))
        pm4py.save_vis_bpmn(heuristic_bpmn, str(heuristic_bpmn_path))
        
        print(f"\nVisualizations saved:")
        print(f"  Petri Net: {heuristic_petri_path}")
        print(f"  BPMN: {heuristic_bpmn_path}")
        
    except Exception as e:
        print(f"Error running Heuristic Miner: {e}")
        import traceback
        traceback.print_exc()
        heuristic_net, heuristic_im, heuristic_fm = None, None, None
        heuristic_bpmn = None
    
    # ============================================================================
    # Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print("Process Discovery Complete")
    print(f"{'='*80}")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nSummary:")
    print(f"  - Alpha Miner: {'Success' if alpha_net is not None else 'Failed'}")
    print(f"  - Inductive Miner: {'Success' if inductive_net is not None else 'Failed'}")
    print(f"  - Heuristic Miner: {'Success' if heuristic_net is not None else 'Failed'}")


if __name__ == "__main__":
    main()

