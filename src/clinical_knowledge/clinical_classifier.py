"""
Stage 2: Clinical Knowledge Layer
Map physiological biomarkers to clinical categories using AHA/ESC guidelines.
"""

from typing import Dict, List, Tuple


def classify_biomarker(metric_name: str, value: float, ranges: Dict) -> Dict[str, str]:
    """
    Classify a single biomarker value against clinical reference ranges.

    Args:
        metric_name: Key in CLINICAL_RANGES config
        value: Measured value
        ranges: CLINICAL_RANGES config dict

    Returns:
        {
            "category": e.g. "normal",
            "risk_level": "healthy"/"caution"/"warning"/"danger",
            "normal_range": "60-100 bpm",
            "suggestion": "..."
        }
    """
    metric_config = ranges.get(metric_name)
    if not metric_config:
        return {"category": "unknown", "risk_level": "unknown",
                "normal_range": "N/A", "suggestion": "No reference data available."}

    risk_emoji = {"healthy": "PASS", "caution": "CHECK", "warning": "WARN", "danger": "ALERT"}

    for category, low, high, risk in metric_config["ranges"]:
        if low <= value < high:
            normal_range_str = _get_normal_range(metric_config["ranges"])
            suggestion = metric_config["suggestion_map"].get(category, "")
            return {
                "category": category,
                "risk_level": risk,
                "risk_flag": risk_emoji.get(risk, "?"),
                "normal_range": f"{normal_range_str} {metric_config['unit']}",
                "suggestion": suggestion,
            }

    return {"category": "out_of_range", "risk_level": "danger",
            "normal_range": "N/A", "suggestion": "Value is outside measurable range."}


def classify_all_biomarkers(biomarkers: Dict[str, float], ranges: Dict) -> List[Dict]:
    """
    Classify all extracted biomarkers.

    Returns:
        List of classification dicts, sorted by risk_level severity.
    """
    severity_order = {"danger": 0, "warning": 1, "caution": 2, "healthy": 3, "unknown": 4}

    results = []
    for metric_name, value in biomarkers.items():
        if metric_name not in ranges:
            continue
        classification = classify_biomarker(metric_name, value, ranges)
        classification["metric_name"] = metric_name
        classification["value"] = value
        classification["unit"] = ranges[metric_name]["unit"]
        results.append(classification)

    # Sort by severity (most concerning first)
    results.sort(key=lambda x: severity_order.get(x["risk_level"], 99))
    return results


def generate_clinical_summary(classifications: List[Dict]) -> str:
    """
    Generate a brief clinical summary string from classifications.
    Used as context for the NL generation layer.
    """
    total = len(classifications)
    healthy = sum(1 for c in classifications if c["risk_level"] == "healthy")
    danger = sum(1 for c in classifications if c["risk_level"] == "danger")
    warning = sum(1 for c in classifications if c["risk_level"] == "warning")
    caution = sum(1 for c in classifications if c["risk_level"] == "caution")

    parts = []
    if danger > 0:
        parts.append(f"{danger} indicator(s) require attention")
    if warning > 0:
        parts.append(f"{warning} indicator(s) are above normal")
    if caution > 0:
        parts.append(f"{caution} indicator(s) borderline")
    if healthy > 0:
        parts.append(f"{healthy}/{total} indicators within healthy range")

    return ", ".join(parts) if parts else "All indicators within normal range"


def _get_normal_range(range_tuples: List[Tuple]) -> str:
    """Extract the normal/healthy range string."""
    for category, low, high, risk in range_tuples:
        if risk == "healthy":
            return f"{low}-{high}"
    return "N/A"
