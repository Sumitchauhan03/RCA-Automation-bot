"""
RCA Automation - Step 2: Sub-reasons analysis for Pro Discarded driver.

Takes a dataset with hub names and sub-reason columns (with weightage percentages).
For a given hub name(s), returns the dominating sub-reasons.
"""

from typing import Optional, Union
import pandas as pd


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_weightage(value) -> float:
    """Parse weightage value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def detect_data_format(df: pd.DataFrame, hub_column: str) -> str:
    """
    Detect if data is in 'long' format (multiple rows per hub) or 'wide' format (one row per hub).
    
    Returns: 'long' or 'wide'
    """
    # Check if hub names repeat (indicates long format)
    hub_counts = df[hub_column].value_counts()
    if len(hub_counts) > 0 and hub_counts.max() > 1:
        return 'long'
    return 'wide'


def find_dominating_sub_reasons(
    df: pd.DataFrame,
    hub_name: str,
    hub_column: str = "hub_name",
    sub_reason_column: Optional[str] = None,
    weightage_column: Optional[str] = None,
    min_weightage_threshold: float = 5.0,
    dominance_ratio: float = 1.5,
    show_all: bool = False,
) -> list[tuple[str, float]]:
    """
    Find dominating sub-reasons for a given hub.
    Supports both long format (multiple rows per hub) and wide format (one row per hub).

    Args:
        df: DataFrame with hub names and sub-reason data.
        hub_name: Name of the hub to analyze.
        hub_column: Name of the column containing hub names.
        sub_reason_column: Name of column with sub-reason names (for long format). Auto-detected if None.
        weightage_column: Name of column with weightage values (for long format). Auto-detected if None.
        min_weightage_threshold: Minimum weightage % to consider (default: 5.0%).
        dominance_ratio: Ratio threshold for determining dominance.
        show_all: If True, show all reasons above threshold. If False, use dominance logic.

    Returns:
        List of tuples: [(sub_reason_name, weightage), ...] sorted by weightage descending.
        Always returns at least one reason (the top one).
    """
    df = _normalize_column_names(df)
    
    if hub_column not in df.columns:
        hub_column = df.columns[0]
    
    # Detect data format
    data_format = detect_data_format(df, hub_column)
    
    sub_reasons = []
    
    if data_format == 'long':
        # Long format: multiple rows per hub, need to filter and extract
        hub_rows = df[df[hub_column].astype(str).str.strip() == str(hub_name).strip()]
        
        if hub_rows.empty:
            return []
        
        # Auto-detect sub_reason_column and weightage_column if not provided
        if sub_reason_column is None:
            # Look for columns that might contain sub-reason names
            # Exclude hub_column and common weightage column names
            possible_cols = [c for c in df.columns if c != hub_column and c.lower() not in ['rl_absolute', 'weightage', 'value', 'percentage', '%']]
            if len(possible_cols) >= 1:
                sub_reason_column = possible_cols[0]  # Usually first non-hub column
            else:
                return []
        
        if weightage_column is None:
            # Look for weightage column (RL_ABSOLUTE, weightage, value, etc.)
            weightage_candidates = [c for c in df.columns if c.lower() in ['rl_absolute', 'weightage', 'value', 'percentage', '%'] or 'absolute' in c.lower()]
            if weightage_candidates:
                weightage_column = weightage_candidates[0]
            else:
                # Try to find numeric column that's not hub_column or sub_reason_column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != hub_column and c != sub_reason_column]
                if numeric_cols:
                    weightage_column = numeric_cols[0]
                else:
                    return []
        
        # Extract sub-reasons and weightages from filtered rows
        for _, row in hub_rows.iterrows():
            sub_reason_name = str(row[sub_reason_column]).strip()
            weightage = _parse_weightage(row[weightage_column])
            if weightage > 0:  # Only include non-zero weightages
                sub_reasons.append((sub_reason_name, weightage))
    
    else:
        # Wide format: one row per hub, columns are sub-reasons
        hub_row = df[df[hub_column].astype(str).str.strip() == str(hub_name).strip()]
        
        if hub_row.empty:
            return []
        
        hub_row = hub_row.iloc[0]
        
        # Get all sub-reason columns (all columns except hub_column)
        sub_reason_cols = [c for c in df.columns if c != hub_column]
        
        # Collect sub-reasons with their weightages
        for col in sub_reason_cols:
            weightage = _parse_weightage(hub_row[col])
            if weightage > 0:  # Only include non-zero weightages
                sub_reasons.append((col, weightage))
    
    if not sub_reasons:
        return []
    
    # Sort by weightage descending
    sub_reasons.sort(key=lambda x: -x[1])
    
    # Identify dominating reasons
    dominating = []
    
    # Always include the top reason (even if below threshold)
    top_reason, top_weightage = sub_reasons[0]
    dominating.append((top_reason, top_weightage))
    
    if show_all:
        # Show all reasons above threshold
        for i in range(1, len(sub_reasons)):
            reason, weightage = sub_reasons[i]
            if weightage >= min_weightage_threshold:
                dominating.append((reason, weightage))
    else:
        # Include all other reasons that are meaningful contributors:
        # 1. Above the minimum threshold (clearly significant), OR
        # 2. Close to the top reason (within dominance_ratio factor) - significant relative contributors, OR
        # 3. Part of a cluster of high reasons (close to previous reason)
        for i in range(1, len(sub_reasons)):
            reason, weightage = sub_reasons[i]
            
            # Include if above threshold
            if weightage >= min_weightage_threshold:
                dominating.append((reason, weightage))
            # Or include if it's a significant contributor relative to top reason
            elif top_weightage > 0 and weightage >= (top_weightage / dominance_ratio):
                dominating.append((reason, weightage))
            # Or include if it's part of a cluster (close to previous reason)
            elif len(dominating) > 0:
                prev_weightage = dominating[-1][1]
                if prev_weightage > 0 and weightage >= (prev_weightage / dominance_ratio):
                    dominating.append((reason, weightage))
    
    return dominating


def analyze_hubs(
    df: pd.DataFrame,
    hub_names: Union[str, list[str]],
    hub_column: str = "hub_name",
    sub_reason_column: Optional[str] = None,
    weightage_column: Optional[str] = None,
    min_weightage_threshold: float = 5.0,
    dominance_ratio: float = 1.5,
    show_all: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    """
    Analyze multiple hubs and return dominating sub-reasons for each.

    Args:
        df: DataFrame with hub names and sub-reason columns.
        hub_names: Single hub name (str) or list of hub names.
        hub_column: Name of the column containing hub names.
        min_weightage_threshold: Minimum weightage % to consider.
        dominance_ratio: Ratio threshold for dominance detection.

    Returns:
        Dictionary: {hub_name: [(sub_reason, weightage), ...], ...}
    """
    if isinstance(hub_names, str):
        hub_names = [hub_names]
    
    results = {}
    for hub_name in hub_names:
        reasons = find_dominating_sub_reasons(
            df, hub_name, hub_column, sub_reason_column, weightage_column,
            min_weightage_threshold, dominance_ratio, show_all
        )
        results[hub_name] = reasons
    
    return results


def format_sub_reasons_output(reasons: list[tuple[str, float]]) -> str:
    """
    Format sub-reasons list as a readable string.

    Args:
        reasons: List of (sub_reason_name, weightage) tuples.

    Returns:
        Formatted string, e.g., "hasAllRequestSkillsV2 (67.9%), hasNoSuitableProfessional (9.2%)"
    """
    if not reasons:
        return "No sub-reasons found"
    
    formatted = []
    for reason, weightage in reasons:
        formatted.append(f"{reason} ({weightage:.1f}%)")
    
    return ", ".join(formatted)


def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load dataframe from CSV file."""
    return pd.read_csv(path)


def load_data_from_string(text: str, sep: Optional[str] = None) -> pd.DataFrame:
    """Load dataframe from pasted text (tab or comma separated)."""
    from io import StringIO
    text = text.strip()
    if sep is None:
        sep = "\t" if "\t" in text.split("\n")[0] else ","
    return pd.read_csv(StringIO(text), sep=sep)


# --- Example usage ---

if __name__ == "__main__":
    # Example: would work with actual CSV data
    print("RCA Step 2: Sub-reasons analysis for Pro Discarded")
    print("Load your CSV and use analyze_hubs() function.")
