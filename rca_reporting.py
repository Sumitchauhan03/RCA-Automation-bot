"""
RCA Automation - Reporting utilities.

Used by Streamlit reporting pages to compute KPIs from the One-shot output and
build rollups (SKU, Pro Discarded reasons).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from rca_step4 import bucketize_statement


CITY_RE = re.compile(r"(?:_hl)?_city_(.+?)_v\d", re.IGNORECASE)


def extract_city(hub_name: str) -> str:
    """
    Extract city from a hub string like:
      ramchandrapuram_city_hyderabad_v2_salon_at_home -> hyderabad
      botanical_garden_hl_city_delhi_v2_salon_at_home -> delhi

    Returns empty string when not detected.
    """
    if not isinstance(hub_name, str):
        return ""
    m = CITY_RE.search(hub_name.lower())
    if not m:
        return ""
    return m.group(1).strip()


@dataclass(frozen=True)
class HubKpis:
    top_pct: float = 0.0
    top_reason: str = ""
    second_pct: float = 0.0
    second_reason: str = ""
    total_top2_pct: float = 0.0


def _parse_reasons(statement: str) -> list[tuple[str, float]]:
    """
    Parse Step 1 style statement into (reason, pct) pairs.
    This duplicates Step 4 extraction to avoid importing private helpers.
    Parentheses are ignored.
    """
    if not isinstance(statement, str):
        return []
    s = statement.strip()
    if not s:
        return []

    # First reason: "X% RL due to reason ..."
    m1 = re.search(r"(\d+\.?\d*)%\s*RL\s*due\s*to\s+(.+?)(?:\s+and\s+|$)", s, re.IGNORECASE)
    out: list[tuple[str, float]] = []
    if m1:
        pct = float(m1.group(1))
        reason_txt = re.sub(r"\([^)]*\)", "", m1.group(2)).strip()
        out.append((reason_txt, pct))

    # Subsequent: "and X% due to reason"
    for m in re.finditer(r"and\s+(\d+\.?\d*)%\s*due\s*to\s+(.+?)(?:\s+and\s+|$)", s, re.IGNORECASE):
        pct = float(m.group(1))
        reason_txt = re.sub(r"\([^)]*\)", "", m.group(2)).strip()
        out.append((reason_txt, pct))

    return out


def _normalize_reason(reason_txt: str) -> str:
    r = (reason_txt or "").strip().lower()
    if not r:
        return ""
    if "supply shortage leave" in r or r == "activity":
        return "Activity"
    if "supply shortage" in r:
        return "Supply shortage"
    if "pro discarded" in r:
        return "Pro discarded"
    return reason_txt.strip().capitalize()


def compute_kpis_from_statement(statement: str) -> HubKpis:
    reasons = [(_normalize_reason(r), pct) for r, pct in _parse_reasons(statement)]
    reasons = [(r, pct) for r, pct in reasons if r and pct is not None]
    if not reasons:
        return HubKpis()

    reasons_sorted = sorted(reasons, key=lambda x: -x[1])
    top_reason, top_pct = reasons_sorted[0]
    second_reason, second_pct = ("", 0.0)
    if len(reasons_sorted) >= 2:
        second_reason, second_pct = reasons_sorted[1]
    return HubKpis(
        top_pct=float(top_pct),
        top_reason=top_reason,
        second_pct=float(second_pct),
        second_reason=second_reason,
        total_top2_pct=float(top_pct) + float(second_pct),
    )


def normalize_oneshot_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming one-shot output DF to expected columns:
      Hub Name | RL RCA | SKU | Summary
    Accepts common variants (e.g., Hubs, RCA, Bucket).
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    colmap = {c.lower(): c for c in cols}

    # Hub column: support HUB_NAME as well
    hub_col = (
        colmap.get("hub name")
        or colmap.get("hub_name")
        or colmap.get("hubs")
        or colmap.get("hub")
        or cols[0]
    )
    rca_col = colmap.get("rl rca") or colmap.get("rca") or colmap.get("statement") or colmap.get("rl_rca")
    # SKU column: handle both 'SKU' and 'SKU RCA' style headers
    sku_col = (
        colmap.get("sku")
        or colmap.get("sku rca")
        or colmap.get("sku_rca")
        or colmap.get("service tags")
        or colmap.get("service_tags")
    )
    summary_col = colmap.get("summary") or colmap.get("bucket")

    out = pd.DataFrame()
    out["Hub Name"] = df[hub_col].astype(str).str.strip()
    out["RL RCA"] = df[rca_col].astype(str) if rca_col else ""
    out["SKU"] = df[sku_col].astype(str) if sku_col else ""
    out["Summary"] = df[summary_col].astype(str) if summary_col else ""
    return out


def add_reporting_fields(df_oneshot: pd.DataFrame, *, bucketize_threshold_ratio: float = 0.5) -> pd.DataFrame:
    """
    Add derived reporting columns:
      city, top_pct, top_reason, second_pct, second_reason, total_top2_pct
    """
    df = normalize_oneshot_columns(df_oneshot)
    df["city"] = df["Hub Name"].apply(extract_city)

    kpis = df["RL RCA"].apply(compute_kpis_from_statement)
    df["top_pct"] = kpis.apply(lambda k: k.top_pct)
    df["top_reason"] = kpis.apply(lambda k: k.top_reason)
    df["second_pct"] = kpis.apply(lambda k: k.second_pct)
    df["second_reason"] = kpis.apply(lambda k: k.second_reason)
    df["total_top2_pct"] = kpis.apply(lambda k: k.total_top2_pct)

    # Ensure Summary exists; if empty, compute from statement
    if "Summary" in df.columns:
        mask = df["Summary"].astype(str).str.strip().eq("")
        if mask.any():
            df.loc[mask, "Summary"] = df.loc[mask, "RL RCA"].apply(
                lambda s: bucketize_statement(str(s), bucketize_threshold_ratio)
            )
    return df


# Matches single 'label (value%)' chunks inside a longer SKU string
SKU_CHUNK_RE = re.compile(r"([^()]+?)\s*\((-?\d+\.?\d*)%\)")


def parse_sku_block(sku_text: str) -> list[tuple[str, float]]:
    """
    Parse the One-shot SKU cell into [(label, rl), ...].

    Supports both:
    - Newline-separated format from rca_step3.format_services_output()
    - Space-separated format like:
        'Pedi (1.75%) Pedi : CrystalRosePedicure (1.29%) ...'
    """
    if not isinstance(sku_text, str) or not sku_text.strip():
        return []
    text = sku_text.strip()
    if text.lower() == "no service tags found":
        return []

    out: list[tuple[str, float]] = []
    for m in SKU_CHUNK_RE.finditer(text):
        label = m.group(1).strip()
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if label:
            out.append((label, val))
    return out


def aggregate_skus(df_reporting: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate SKU RL across the provided reporting dataframe.
    Returns: DataFrame(label, rl_sum, hubs_count)
    """
    rows: list[dict[str, object]] = []
    for _, r in df_reporting.iterrows():
        hub = str(r.get("Hub Name", "")).strip()
        for label, val in parse_sku_block(str(r.get("SKU", ""))):
            rows.append({"label": label, "rl": float(val), "hub": hub})

    if not rows:
        return pd.DataFrame(columns=["label", "rl_sum", "hubs_count"])

    tmp = pd.DataFrame(rows)
    agg = (
        tmp.groupby("label", as_index=False)
        .agg(rl_sum=("rl", "sum"), hubs_count=("hub", "nunique"))
        .sort_values(["rl_sum", "hubs_count"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return agg


PRO_DISCARD_RE = re.compile(r"pro discarded\s*\(([^)]*)\)", re.IGNORECASE)


def extract_pro_discard_reasons(rl_rca_text: str) -> list[str]:
    """
    Extract the injected Pro Discarded reason list from RL RCA text:
      '... pro discarded (a, b & c) ...' -> ['a', 'b', 'c']

    Returns empty list if not present.
    """
    if not isinstance(rl_rca_text, str):
        return []
    m = PRO_DISCARD_RE.search(rl_rca_text)
    if not m:
        return []
    chunk = m.group(1).strip()
    if not chunk:
        return []
    parts = re.split(r"\s*&\s*|\s*,\s*", chunk)
    return [p.strip() for p in parts if p and p.strip()]


def pro_discard_rollup(df_reporting: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, r in df_reporting.iterrows():
        hub = str(r.get("Hub Name", "")).strip()
        for reason in extract_pro_discard_reasons(str(r.get("RL RCA", ""))):
            rows.append({"reason": reason, "hub": hub})
    if not rows:
        return pd.DataFrame(columns=["reason", "hubs_count"])
    tmp = pd.DataFrame(rows)
    return (
        tmp.groupby("reason", as_index=False)
        .agg(hubs_count=("hub", "nunique"))
        .sort_values(["hubs_count", "reason"], ascending=[False, True])
        .reset_index(drop=True)
    )

