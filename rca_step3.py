"""
RCA Automation - Step 3: Service / Tag analysis for RL.

Takes a long-format dataset with hub names, category, tags, and RL weightage.
For a given hub name(s), returns dominating service tags that drive RL.
"""

from typing import Optional, Union

import pandas as pd


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_service_tag(tags: str) -> tuple[str, Optional[str]]:
    """
    Parse a tags string like:
      "pedicure::gender:female::product_used_skill_ind:CrystalRosePedicure"
    into:
      service = "Pedi" (or capitalized service name)
      detail  = "CrystalRosePedicure" (product / tool / SOP etc.)
    """
    if not isinstance(tags, str):
        return "", None

    parts = [p.strip() for p in tags.split("::") if p.strip()]
    if not parts:
        return "", None

    raw_service = parts[0].lower()

    # Map raw service to a nicer label
    service_map = {
        "pedicure": "Pedi",
        "manicure": "Mani",
        "waxing": "Waxing",
        "hair_color": "Hair Color",
        "hair_spa": "Hair Spa",
        "hair_oil_massage": "Hair Oil Massage",
        "facial": "Facial",
        "cleanup": "Cleanup",
        "threading": "Threading",
        "face_waxing": "Face Waxing",
        "bikini_waxing": "Bikini Waxing",
    }
    service = service_map.get(raw_service, raw_service.capitalize())

    detail_keys = [
        "product_used_skill_ind",
        "tool_used_skill_ind",
        "sop_skill_ind",
        "sop_ind",
    ]
    detail: Optional[str] = None

    for part in parts[1:]:
        if ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key in detail_keys and val:
            detail = val
            break

    return service, detail


def aggregate_tags_for_hub(
    df: pd.DataFrame,
    hub_name: str,
    hub_column: str = "Hub Name",
    tags_column: str = "Tags",
    rl_column: str = "RL",
    min_rl_threshold: float = 0.0,
    dominance_ratio: float = 1.5,
    show_all: bool = False,
) -> list[tuple[str, float]]:
    """
    Aggregate RL by parsed service tags for a given hub.

    Returns list of (label, rl_sum) sorted by rl_sum desc.
    Label format example: "Pedi : CrystalRosePedicure" or "Pedi : heel_peel".
    """
    df = _normalize_column_names(df)

    # Normalize column names for matching
    hub_col = hub_column
    tags_col = tags_column
    rl_col = rl_column

    # Fallbacks if exact names not present
    col_map = {c.lower(): c for c in df.columns}
    if hub_col not in df.columns:
        hub_col = col_map.get("hub name".lower(), df.columns[0])
    if tags_col not in df.columns:
        tags_col = col_map.get("tags", df.columns[1] if len(df.columns) > 1 else df.columns[0])
    if rl_col not in df.columns:
        for cand in ["rl", "rl_absolute", "weightage", "value", "percentage"]:
            if cand in col_map:
                rl_col = col_map[cand]
                break

    hub_rows = df[df[hub_col].astype(str).str.strip() == str(hub_name).strip()]
    if hub_rows.empty:
        return []

    agg: dict[str, float] = {}

    for _, row in hub_rows.iterrows():
        tags_val = row[tags_col]
        rl_val = _parse_float(row[rl_col])
        if rl_val == 0:
            continue

        service, detail = _parse_service_tag(str(tags_val))
        if not service:
            continue

        if detail:
            label = f"{service} : {detail}"
        else:
            label = service

        agg[label] = agg.get(label, 0.0) + rl_val

    if not agg:
        return []

    items = sorted(agg.items(), key=lambda x: -x[1])

    # Identify dominating tags
    dominating: list[tuple[str, float]] = []
    top_label, top_rl = items[0]
    dominating.append((top_label, top_rl))

    if show_all:
        # Show all tags that have any RL contribution (> 0), no additional filtering
        for label, rl_sum in items[1:]:
            if rl_sum > 0:
                dominating.append((label, rl_sum))
    else:
        # Focus on dominant tags while still using full data for aggregation
        for label, rl_sum in items[1:]:
            # Ignore only truly negligible tags (below min_rl_threshold), but
            # all rows were already used for aggregation above.
            if rl_sum >= min_rl_threshold:
                dominating.append((label, rl_sum))
            # Or include if it's significant relative to top tag
            elif top_rl > 0 and rl_sum >= (top_rl / dominance_ratio):
                dominating.append((label, rl_sum))

    return dominating


def analyze_hubs_services(
    df: pd.DataFrame,
    hub_names: Union[str, list[str]],
    hub_column: str = "Hub Name",
    tags_column: str = "Tags",
    rl_column: str = "RL",
    min_rl_threshold: float = 0.0,
    dominance_ratio: float = 1.5,
    show_all: bool = False,
) -> dict[str, list[tuple[str, float]]]:
    """
    Analyze multiple hubs and return dominating service tags for each.
    """
    if isinstance(hub_names, str):
        hub_names = [hub_names]

    df = _normalize_column_names(df)
    results: dict[str, list[tuple[str, float]]] = {}
    for hub in hub_names:
        reasons = aggregate_tags_for_hub(
            df,
            hub,
            hub_column=hub_column,
            tags_column=tags_column,
            rl_column=rl_column,
            min_rl_threshold=min_rl_threshold,
            dominance_ratio=dominance_ratio,
            show_all=show_all,
        )
        results[hub] = reasons
    return results


def format_services_output(services: list[tuple[str, float]]) -> str:
    """
    Format list of (label, rl_sum) into a newline-separated string.

    Output example:
      Pedi : CrystalRosePedicure (0.676%)
      Waxing : RicaWhiteChocolateWax (0.231%)
    """
    if not services:
        return "No service tags found"

    def _fmt_rl(x: float) -> str:
        # Keep small values readable (e.g., 0.676) without forcing extra zeros
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return s if s else "0"

    # One tag per line for readability / copy-paste
    lines: list[str] = []
    for label, rl_sum in services:
        lines.append(f"{label} ({_fmt_rl(rl_sum)}%)")
    return "\n".join(lines)


if __name__ == "__main__":
    print("RCA Step 3: Service / Tag analysis for RL")
    print("Load your CSV and use analyze_hubs_services() function.")

