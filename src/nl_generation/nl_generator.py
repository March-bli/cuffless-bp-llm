"""
Stage 3: Natural Language Generation Layer
Rule-based template NL generation from classified physiological biomarkers.
"""

from typing import Dict, List
from src.utils.config import METRIC_NAMES_DISPLAY, INTRO_TEMPLATE, OUTRO_TEMPLATE


def generate_single_metric_nl(classification: Dict) -> str:
    """
    Generate natural language description for a single biomarker.

    Template structure:
    "{DisplayName} is {value} {unit}. This falls within the {range_name} category
    ({normal_range}). {suggestion}"
    """
    metric = classification["metric_name"]
    display_name = METRIC_NAMES_DISPLAY.get(metric, metric)
    value = classification["value"]
    unit = classification["unit"]
    normal_range = classification["normal_range"]
    suggestion = classification.get("suggestion", "")

    # Build description
    lines = []

    # Main measurement line
    lines.append(f"{display_name}: {value} {unit}.")

    # Classification line
    category = classification["category"].replace("_", " ")
    lines.append(f"  Status: {category}. (Reference range: {normal_range})")

    # Suggestion if not healthy
    if classification["risk_level"] != "healthy" and suggestion:
        lines.append(f"  Note: {suggestion}")

    return "\n".join(lines)


def generate_health_report(
    classifications: List[Dict],
    signal_quality: Dict,
    duration_sec: int = 10,
) -> str:
    """
    Generate a complete natural language health report from classifications.

    Args:
        classifications: List of classified biomarkers from Stage 2
        signal_quality: Signal quality assessment from Stage 1
        duration_sec: Duration of the measurement

    Returns:
        Full NL health report string
    """
    lines = []

    # Intro
    lines.append(INTRO_TEMPLATE.format(duration=duration_sec))
    lines.append("=" * 60)

    # Signal quality note
    quality_label = signal_quality.get("quality_label", "unknown")
    if quality_label == "poor":
        lines.append("\n[NOTE: Signal quality was poor. Measurements may be less accurate.]\n")
    elif quality_label == "acceptable":
        lines.append("\n[Signal quality was acceptable.]\n")

    # Severity-grouped output
    severity_groups = _group_by_risk(classifications)
    risk_section_labels = {
        "danger": "--- Indicators Requiring Attention ---",
        "warning": "--- Indicators to Monitor ---",
        "caution": "--- Borderline Indicators ---",
        "healthy": "--- Healthy Indicators ---",
    }

    for risk_level in ["danger", "warning", "caution", "healthy"]:
        items = severity_groups.get(risk_level, [])
        if not items:
            continue
        lines.append(risk_section_labels.get(risk_level, ""))
        for c in items:
            lines.append(generate_single_metric_nl(c))
        lines.append("")

    # Overall assessment
    lines.append("--- Overall Assessment ---")
    danger_count = len(severity_groups.get("danger", []))
    warning_count = len(severity_groups.get("warning", []))
    caution_count = len(severity_groups.get("caution", []))
    healthy_count = len(severity_groups.get("healthy", []))

    if danger_count > 0:
        lines.append(f"{danger_count} indicator(s) raised concern. Medical consultation is recommended.")
    elif warning_count > 0:
        lines.append(f"Your health indicators are generally stable, with {warning_count} above normal range.")
    elif caution_count > 0:
        lines.append(f"Most indicators are fine. {caution_count} indicator(s) are near boundary — worth monitoring.")
    else:
        lines.append("All measured indicators are within healthy ranges. Keep up your good health habits!")

    # Outro disclaimer
    lines.append(OUTRO_TEMPLATE)

    return "\n".join(lines)


def generate_compact_nl(classifications: List[Dict], signal_quality: Dict) -> str:
    """
    Compact single-paragraph NL summary suitable for chat/voice output.
    For the AI assistant integration on the embedded board.
    """
    healthy_items = []
    concern_items = []

    for c in classifications:
        metric = c["metric_name"]
        display = METRIC_NAMES_DISPLAY.get(metric, metric)
        value = c["value"]
        unit = c["unit"]

        if c["risk_level"] == "healthy":
            healthy_items.append(f"{display} is {value} {unit} (healthy)")
        else:
            concern_items.append(
                f"{display} is {value} {unit} — {c['category'].replace('_', ' ')}"
            )

    parts = []
    if concern_items:
        parts.append("Attention: " + ". ".join(concern_items) + ".")
    if healthy_items:
        parts.append("Good: " + ". ".join(healthy_items) + ".")

    if not parts:
        return "No health data available."

    quality_prefix = ""
    if signal_quality.get("quality_label") == "poor":
        quality_prefix = "[Low signal quality — readings approximate.] "

    return quality_prefix + " ".join(parts) + " This is not medical advice."


def _group_by_risk(classifications: List[Dict]) -> Dict[str, List[Dict]]:
    """Group classifications by risk level."""
    groups = {"danger": [], "warning": [], "caution": [], "healthy": []}
    for c in classifications:
        risk = c.get("risk_level", "unknown")
        if risk in groups:
            groups[risk].append(c)
    return groups
