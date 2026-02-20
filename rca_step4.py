"""
RCA Automation - Step 4: Summary / Bucketization step.

Takes RCA statements like "8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage"
and bucketizes them into predefined categories.
"""

import re
from typing import Optional


def _parse_percentage(text: str) -> Optional[float]:
    """Extract percentage value from text like '8%' or '8.5%'."""
    match = re.search(r'(\d+\.?\d*)%', text)
    if match:
        return float(match.group(1))
    return None


def _normalize_reason_name(reason: str) -> str:
    """
    Normalize reason names:
    - "supply shortage leave" or "supply shortage leave (activity)" → "Activity"
    - "supply shortage" → "Supply shortage"
    - "pro discarded" → "Pro discarded"
    """
    reason_lower = reason.lower().strip()
    
    # Check for "supply shortage leave" or "supply shortage leave (activity)"
    if "supply shortage leave" in reason_lower or reason_lower == "activity":
        return "Activity"
    
    # Check for "supply shortage"
    if "supply shortage" in reason_lower:
        return "Supply shortage"
    
    # Check for "pro discarded"
    if "pro discarded" in reason_lower:
        return "Pro discarded"
    
    # Default: return capitalized
    return reason.strip().capitalize()


def _extract_reasons_from_statement(statement: str) -> list[tuple[str, float]]:
    """
    Parse a statement like:
    "8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage"
    
    Returns: [(reason_name, percentage), ...]
    """
    reasons = []
    
    # Pattern 1: "X% RL due to [reason]" (first reason)
    pattern1 = r'(\d+\.?\d*)%\s*RL\s*due\s*to\s+(.+?)(?:\s+and\s+|$)'
    match1 = re.search(pattern1, statement, re.IGNORECASE)
    
    if match1:
        pct_str = match1.group(1)
        reason_text = match1.group(2).strip()
        
        # Remove parentheses content (like "(hasAllRequestSkillsV2)")
        reason_text = re.sub(r'\([^)]*\)', '', reason_text).strip()
        
        pct = float(pct_str)
        normalized_reason = _normalize_reason_name(reason_text)
        reasons.append((normalized_reason, pct))
    
    # Pattern 2: "and X% due to [reason]" (subsequent reasons)
    pattern2 = r'and\s+(\d+\.?\d*)%\s*due\s*to\s+(.+?)(?:\s+and\s+|$)'
    matches2 = re.finditer(pattern2, statement, re.IGNORECASE)
    
    for match in matches2:
        pct_str = match.group(1)
        reason_text = match.group(2).strip()
        
        # Remove parentheses content
        reason_text = re.sub(r'\([^)]*\)', '', reason_text).strip()
        
        pct = float(pct_str)
        normalized_reason = _normalize_reason_name(reason_text)
        reasons.append((normalized_reason, pct))
    
    return reasons


def _should_ignore_smaller_reason(reasons: list[tuple[str, float]], threshold_ratio: float = 0.5) -> list[tuple[str, float]]:
    """
    If difference between reasons is much high, ignore the smaller one.
    Example: 16% supply shortage and 3% pro discarded → ignore pro discarded.
    
    threshold_ratio: if smaller reason < (larger reason * ratio), ignore it.
    """
    if len(reasons) < 2:
        return reasons
    
    # Sort by percentage descending
    sorted_reasons = sorted(reasons, key=lambda x: -x[1])
    
    largest_pct = sorted_reasons[0][1]
    filtered = [sorted_reasons[0]]  # Always keep the largest
    
    for reason, pct in sorted_reasons[1:]:
        # Keep if it's significant relative to largest (above threshold)
        if largest_pct > 0 and pct >= (largest_pct * threshold_ratio):
            filtered.append((reason, pct))
    
    return filtered


def _bucketize_reasons(reasons: list[tuple[str, float]]) -> str:
    """
    Map reasons to one of the predefined buckets.
    
    Buckets:
    - Activity
    - Activity + Pro discarded 
    - Activity + Supply shortage
    - Activity + Supply shortage + Pro discarded
    - Pro discarded
    - Pro discarded + Activity
    - Pro discarded + Supply shortage
    - Supply shortage
    - Supply shortage + Activity
    - Supply shortage + Pro discarded
    """
    if not reasons:
        return ""
    
    # Get unique reason names (ignore percentages for bucketization)
    reason_names = sorted(set(r[0] for r in reasons))
    
    # Map to bucket
    if len(reason_names) == 1:
        return reason_names[0]
    
    # Two reasons
    if len(reason_names) == 2:
        r1, r2 = reason_names
        
        # Check all combinations
        if r1 == "Activity" and r2 == "Pro discarded":
            return "Activity + Pro discarded"
        elif r1 == "Activity" and r2 == "Supply shortage":
            return "Activity + Supply shortage"
        elif r1 == "Pro discarded" and r2 == "Activity":
            return "Pro discarded + Activity"
        elif r1 == "Pro discarded" and r2 == "Supply shortage":
            return "Pro discarded + Supply shortage"
        elif r1 == "Supply shortage" and r2 == "Activity":
            return "Supply shortage + Activity"
        elif r1 == "Supply shortage" and r2 == "Pro discarded":
            return "Supply shortage + Pro discarded"
        else:
            # Fallback: join with +
            return " + ".join(reason_names)
    
    # Three reasons
    if len(reason_names) == 3:
        if "Activity" in reason_names and "Pro discarded" in reason_names and "Supply shortage" in reason_names:
            return "Activity + Supply shortage + Pro discarded"
        else:
            # Fallback: join with +
            return " + ".join(reason_names)
    
    # More than 3: join all
    return " + ".join(reason_names)


def bucketize_statement(statement: str, threshold_ratio: float = 0.5) -> str:
    """
    Parse a single RCA statement and return its bucketized summary.
    
    Args:
        statement: RCA statement like "8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage"
        threshold_ratio: Ratio threshold for ignoring smaller reasons (default: 0.5 = 50%)
    
    Returns:
        Bucketized summary like "Pro discarded + Supply shortage"
    """
    statement = statement.strip()
    if not statement:
        return ""
    
    # Extract reasons with percentages
    reasons = _extract_reasons_from_statement(statement)
    
    if not reasons:
        return ""
    
    # Filter out reasons that are too small relative to the largest
    filtered_reasons = _should_ignore_smaller_reason(reasons, threshold_ratio)
    
    # Bucketize
    bucket = _bucketize_reasons(filtered_reasons)
    
    return bucket


def bucketize_statements(statements: list[str], threshold_ratio: float = 0.5) -> list[str]:
    """
    Process multiple RCA statements and return bucketized summaries.
    
    Args:
        statements: List of RCA statements
        threshold_ratio: Ratio threshold for ignoring smaller reasons
    
    Returns:
        List of bucketized summaries
    """
    results = []
    for statement in statements:
        bucket = bucketize_statement(statement, threshold_ratio)
        results.append(bucket)
    return results


if __name__ == "__main__":
    # Test examples
    test_statements = [
        "8% RL due to pro discarded (hasAllRequestSkillsV2) and 6% due to supply shortage",
        "16% RL due to supply shortage and 3% due to pro discarded",
        "7% RL due to supply shortage",
        "5% RL due to supply shortage leave (activity) and 4% due to pro discarded",
    ]
    
    print("Testing Step 4: Statement Bucketization")
    print("=" * 60)
    
    for stmt in test_statements:
        bucket = bucketize_statement(stmt)
        print(f"\nStatement: {stmt}")
        print(f"Bucket: {bucket}")
