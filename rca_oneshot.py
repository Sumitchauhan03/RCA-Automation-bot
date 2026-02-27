"""
RCA Automation - One-shot computation helpers.

This module centralizes the "one-shot RCA" pipeline so it can be reused by:
- The One-shot RCA UI
- Reporting dashboards (when raw Step 1/2/3 datasets are provided)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from rca_step1 import add_rca_column
from rca_step2 import analyze_hubs as analyze_hubs_step2
from rca_step3 import analyze_hubs_services, format_services_output
from rca_step4 import bucketize_statement


@dataclass(frozen=True)
class OneShotSettings:
    step2_min_weightage: float = 5.0
    step2_dominance_ratio: float = 1.5
    step3_min_rl: float = 0.0
    step3_dominance_ratio: float = 1.5
    step4_threshold_ratio: float = 0.5


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _format_sub_reason_list(reasons: list[tuple[str, float]]) -> str:
    if not reasons:
        return ""
    names = [r for r, _ in reasons]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    return ", ".join(names[:-1]) + f" & {names[-1]}"


def _inject_pro_discarded_details(rca_text: str, reasons: list[tuple[str, float]]) -> str:
    base = rca_text or ""
    details = _format_sub_reason_list(reasons)
    if not base or not details:
        return (base or "").replace("supply shortage leave (activity)", "activity")
    if "pro discarded" in base:
        base = base.replace("pro discarded", f"pro discarded ({details})", 1)
    base = base.replace("supply shortage leave (activity)", "activity")
    return base


def _detect_step3_columns(df3: pd.DataFrame) -> tuple[str, str, str]:
    cols3 = list(df3.columns)
    colmap3 = {str(c).strip().lower(): c for c in cols3}

    if "hub name" in colmap3:
        hub_col3 = colmap3["hub name"]
    elif "hubs" in colmap3:
        hub_col3 = colmap3["hubs"]
    elif "hub" in colmap3:
        hub_col3 = colmap3["hub"]
    else:
        hub_col3 = cols3[0]

    if "tags" in colmap3:
        tags_col3 = colmap3["tags"]
    elif "tag" in colmap3:
        tags_col3 = colmap3["tag"]
    else:
        tags_col3 = cols3[1] if len(cols3) > 1 else cols3[0]

    rl_col3: Optional[str] = None
    for cand in ["rl", "rl_absolute", "weightage", "value", "percentage"]:
        if cand in colmap3:
            rl_col3 = colmap3[cand]
            break
    if rl_col3 is None:
        rl_col3 = cols3[-1]

    return hub_col3, tags_col3, rl_col3


def compute_one_shot(
    df_step1: pd.DataFrame,
    df_step2: pd.DataFrame,
    df_step3: pd.DataFrame,
    hub_names: Optional[list[str]] = None,
    *,
    hub_col_step1: Optional[str] = None,
    driver_cols_step1: Optional[list[str]] = None,
    settings: OneShotSettings = OneShotSettings(),
) -> pd.DataFrame:
    """
    Compute the One-shot RCA output table:
      Hub Name | RL RCA | SKU | Summary

    Notes:
    - Summary bucketization uses the Step 1 RCA statement (without sub-reason details).
    - RL RCA injects Pro Discarded sub-reason names in parentheses (from Step 2).
    - SKU is produced from Step 3 formatted services output.
    """
    df1 = _normalize_columns(df_step1)
    df2 = _normalize_columns(df_step2)
    df3 = _normalize_columns(df_step3)

    cols1 = list(df1.columns)
    hub_col1 = hub_col_step1 or cols1[0]
    if driver_cols_step1 is None:
        driver_cols1 = cols1[1:4] if len(cols1) >= 4 else cols1[1:]
    else:
        driver_cols1 = driver_cols_step1

    if hub_names is None:
        hub_names = [str(x).strip() for x in df1[hub_col1].tolist() if str(x).strip()]

    df1_with_rca = add_rca_column(df1, hub_col1, driver_cols1, None)

    rca_map_step1: dict[str, str] = {}
    for _, row in df1_with_rca.iterrows():
        hub_val = str(row[hub_col1]).strip()
        rca_map_step1[hub_val] = str(row.get("RCA", ""))

    sub_reason_results = analyze_hubs_step2(
        df2,
        hub_names,
        hub_column=df2.columns[0],
        sub_reason_column=None,
        weightage_column=None,
        min_weightage_threshold=settings.step2_min_weightage,
        dominance_ratio=settings.step2_dominance_ratio,
        show_all=False,
    )

    hub_col3, tags_col3, rl_col3 = _detect_step3_columns(df3)
    services_results = analyze_hubs_services(
        df3,
        hub_names,
        hub_column=hub_col3,
        tags_column=tags_col3,
        rl_column=rl_col3,
        min_rl_threshold=settings.step3_min_rl,
        dominance_ratio=settings.step3_dominance_ratio,
        show_all=False,
    )

    rows: list[dict[str, str]] = []
    for hub in hub_names:
        hub_key = str(hub).strip()
        base_rca = rca_map_step1.get(hub_key, "")
        sub_reasons = sub_reason_results.get(hub_key, [])
        services = services_results.get(hub_key, [])

        rca_with_details = _inject_pro_discarded_details(base_rca, sub_reasons)
        sku_text = format_services_output(services)
        summary_bucket = bucketize_statement(base_rca, settings.step4_threshold_ratio) if base_rca else ""

        rows.append(
            {
                "Hub Name": hub_key,
                "RL RCA": rca_with_details,
                "SKU": sku_text,
                "Summary": summary_bucket,
            }
        )

    return pd.DataFrame(rows)

