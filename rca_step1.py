"""
RCA Automation - Step 1: Hub-wise RCA from RL% drivers.

Takes a dataset with hub names and 3 driver columns (metrics).
Outputs an RCA column: for each hub, the top 2 drivers with rounded %
in format: "X% RL due to [metric1] and Y% due to [metric2]".

Input: CSV file, file path, or pasted tab/coma-separated data.
"""

from io import StringIO
from typing import Optional

import pandas as pd


# Default display names for metrics (column name -> output label)
DEFAULT_DISPLAY_NAMES = {
    "Supply Shortage (Leave)": "supply shortage leave (activity)",
    "supply_shortage": "supply shortage",
    "Pro Discarded": "pro discarded",
}


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _display_name(col: str, custom_display: Optional[dict] = None) -> str:
    """Get display name for a driver column (lowercase, human-readable)."""
    mapping = {**(custom_display or {}), **DEFAULT_DISPLAY_NAMES}
    if col in mapping:
        return mapping[col].strip().lower()
    # Fallback: lowercase, replace underscores with spaces
    return col.strip().lower().replace("_", " ")


def _round_pct(value) -> int:
    """Round to integer percentage (4.8 -> 5, 4.5 -> 5, 4.4 -> 4). Round half up."""
    try:
        x = float(value)
        return int(x + 0.5) if x >= 0 else int(x - 0.5)
    except (TypeError, ValueError):
        return 0


def compute_rca_row(
    row: pd.Series,
    driver_columns: list[str],
    hub_column: str,
    display_names: Optional[dict] = None,
) -> str:
    """
    For one hub row: pick top 2 drivers by value, round, format RCA string.
    """
    values = []
    for col in driver_columns:
        if col not in row.index:
            continue
        val = row[col]
        pct = _round_pct(val)
        if pct < 0:
            pct = 0
        label = _display_name(col, display_names)
        values.append((pct, label, col))

    if len(values) < 2:
        return ""

    # Sort by rounded % descending, take top 2
    values.sort(key=lambda x: (-x[0], x[1]))
    top2 = values[:2]

    # Format: "X% RL due to [metric1] and Y% due to [metric2]"
    first_pct, first_label, _ = top2[0]
    second_pct, second_label, _ = top2[1]
    
    # If second driver is very less than first (less than 50% of first), ignore it
    if first_pct > 0 and second_pct < (first_pct * 0.5):
        return f"{first_pct}% RL due to {first_label}"
    
    return f"{first_pct}% RL due to {first_label} and {second_pct}% due to {second_label}"


def add_rca_column(
    df: pd.DataFrame,
    hub_column: str,
    driver_columns: list[str],
    display_names: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Add 'RCA' column to the dataframe.
    """
    df = _normalize_column_names(df)
    if hub_column not in df.columns:
        hub_column = df.columns[0]

    rca_values = df.apply(
        lambda row: compute_rca_row(row, driver_columns, hub_column, display_names),
        axis=1,
    )
    out = df.copy()
    out["RCA"] = rca_values
    return out


def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load dataframe from CSV file."""
    return pd.read_csv(path)


def load_data_from_string(
    text: str, sep: Optional[str] = None
) -> pd.DataFrame:
    """
    Load dataframe from pasted text (tab or comma separated).
    Auto-detects separator if not provided.
    """
    text = text.strip()
    if sep is None:
        sep = "\t" if "\t" in text.split("\n")[0] else ","
    return pd.read_csv(StringIO(text), sep=sep)


def run_rca(
    data_source: str,
    hub_column: str,
    driver_columns: list[str],
    driver_weights: Optional[list[float]] = None,
    display_names: Optional[dict] = None,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run RCA step 1.

    Args:
        data_source: Path to CSV file, or raw string (tab/comma separated).
        hub_column: Name of the column containing hub names.
        driver_columns: List of exactly 3 driver column names.
        driver_weights: Optional weightage for each driver (for future use).
        display_names: Optional mapping {column_name: "display name"}.
        output_path: If set, save result CSV here.

    Returns:
        DataFrame with 'RCA' column added.
    """
    # Load data
    if data_source.endswith(".csv") or "\n" not in data_source.strip():
        try:
            df = load_data_from_csv(data_source)
        except Exception:
            df = load_data_from_string(data_source)
    else:
        df = load_data_from_string(data_source)

    df = _normalize_column_names(df)

    if len(driver_columns) != 3:
        raise ValueError("Exactly 3 driver columns are required.")

    missing = [c for c in driver_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Driver columns not found in data: {missing}")

    result = add_rca_column(df, hub_column, driver_columns, display_names)

    if output_path:
        result.to_csv(output_path, index=False)

    return result


def print_rca_report(df: pd.DataFrame, hub_column: Optional[str] = None) -> None:
    """Print RCA in the format: hub_name then RCA line, blank line."""
    df = _normalize_column_names(df)
    hub_col = hub_column or (df.columns[0] if "RCA" in df.columns else "hub_name_2")
    if hub_col not in df.columns:
        hub_col = df.columns[0]
    for _, row in df.iterrows():
        print(row[hub_col])
        print(row["RCA"])
        print()


# --- CLI and example usage ---

if __name__ == "__main__":
    import sys

    # Example: pasted data
    EXAMPLE_DATA = """hub_name_2	Pro Discarded	Supply Shortage (Leave)	supply_shortage
ameenpur_hl_city_hyderabad_v2_salon_at_home	1.6	1.3	6.7
ameerpet_city_hyderabad_v2_salon_at_home	3.3	1.2	6.4
bachupally_hl_city_hyderabad_v2_salon_at_home	2.7	1.3	4.5"""

    HUB_COL = "hub_name_2"
    DRIVERS = ["Pro Discarded", "Supply Shortage (Leave)", "supply_shortage"]

    if len(sys.argv) >= 2:
        # CSV file path provided
        csv_path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
        result = run_rca(csv_path, HUB_COL, DRIVERS, output_path=out_path)
    else:
        # Use example pasted data
        result = run_rca(EXAMPLE_DATA, HUB_COL, DRIVERS)

    print_rca_report(result, HUB_COL)
    print("--- DataFrame with RCA column ---")
    print(result.to_string())
