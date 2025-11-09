def print_event_log_columns_and_values(log_path: str) -> None:
    """
    Load an event log from an XES file and print the available columns together with their values.
    """
    log = pm4py.read_xes(log_path)
    df = pm4py.convert_to_dataframe(log)

    print("Event log columns:")
    for column in df.columns:
        print(f"- {column}")

    print("\nColumn values:")
    for column in df.columns:
        values = df[column].dropna().unique()
        print(f"\n{column}:")
        if len(values) > 20:
            preview = ", ".join(map(str, values[:20]))
            print(f"{preview}, ... ({len(values)} unique values total)")
        else:
            joined_values = ", ".join(map(str, values))
            print(joined_values if joined_values else "No values (all NaN)")


if __name__ == "__main__":
    # Update the path below if your log file is located elsewhere.
    log_file_path = r"Dataset\BPI Challenge 2017.xes"
    print_event_log_columns_and_values(log_file_path)