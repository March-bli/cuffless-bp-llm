"""
LLM-based Natural Language Health Report Generator.
Supports multiple backends: xAI (Grok), DeepSeek, OpenAI-compatible APIs.

Replaces the rule-based template system with LLM-generated,
clinically-informed, natural-sounding health summaries.
"""

import json
import os
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass


# ── backend configs ────────────────────────────────────────────────────────
@dataclass
class LLMBackend:
    """Configuration for an LLM API backend."""
    name: str
    api_base: str
    model: str
    api_key: str

    def chat(self, messages: List[Dict], temperature: float = 0.3,
             max_tokens: int = 800) -> str:
        """Send a chat completion request."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers, json=payload, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ── backend factory ─────────────────────────────────────────────────────────
def get_backend(provider: str = "auto") -> LLMBackend:
    """Auto-detect or select an LLM backend.

    Priority: env vars → provider arg → fallback
    """
    # xAI / Grok
    xai_key = os.environ.get("XAI_API_KEY", "")
    if (provider in ("auto", "xai", "grok")) and xai_key:
        return LLMBackend(
            name="xai",
            api_base="https://api.x.ai/v1",
            model="grok-2-latest",  # fast, cheap, good enough for reports
            api_key=xai_key,
        )

    # DeepSeek
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if (provider in ("auto", "deepseek")) and ds_key:
        return LLMBackend(
            name="deepseek",
            api_base="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key=ds_key,
        )

    raise RuntimeError(
        "No LLM API key found. Set XAI_API_KEY or DEEPSEEK_API_KEY env var."
    )


# ── prompt builder ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a professional health report assistant. Based on physiological 
measurements, generate a concise, natural-language health summary.

Guidelines:
- Write in {language}, use plain, empathetic language.
- Highlight concerning indicators first, then mention healthy ones.
- Include specific numbers with units, and explain what they mean in everyday terms.
- NEVER claim this is a medical diagnosis. Always include a disclaimer.
- Keep it under 200 words. Be warm but professional.
- Use bullet points only for the summary section.
"""


def _build_prompt(
    classifications: List[Dict],
    signal_quality: Dict,
    language: str = "Chinese",
) -> List[Dict]:
    """Build the chat messages for LLM health report generation."""

    # Organize data for the prompt
    danger_items = []
    warning_items = []
    caution_items = []
    healthy_items = []

    for c in classifications:
        entry = {
            "metric": c["metric_name"],
            "value": c["value"],
            "unit": c["unit"],
            "category": c["category"].replace("_", " "),
            "risk": c["risk_level"],
            "normal": c["normal_range"],
        }
        if c["risk_level"] == "danger":
            danger_items.append(entry)
        elif c["risk_level"] == "warning":
            warning_items.append(entry)
        elif c["risk_level"] == "caution":
            caution_items.append(entry)
        else:
            healthy_items.append(entry)

    # Build the user prompt
    parts = ["Here are the measurement results:\n"]

    if danger_items:
        parts.append("⚠️ CRITICAL indicators:")
        for item in danger_items:
            parts.append(
                f"  - {item['metric']}: {item['value']} {item['unit']} "
                f"({item['category']}, normal: {item['normal']})"
            )

    if warning_items:
        parts.append("\n⚠️ WARNING indicators:")
        for item in warning_items:
            parts.append(
                f"  - {item['metric']}: {item['value']} {item['unit']} "
                f"({item['category']}, normal: {item['normal']})"
            )

    if caution_items:
        parts.append("\n⚡ BORDERLINE indicators:")
        for item in caution_items:
            parts.append(
                f"  - {item['metric']}: {item['value']} {item['unit']} "
                f"({item['category']}, normal: {item['normal']})"
            )

    if healthy_items:
        parts.append("\n✅ HEALTHY indicators:")
        for item in healthy_items:
            parts.append(
                f"  - {item['metric']}: {item['value']} {item['unit']} "
                f"({item['category']})"
            )

    # Signal quality note
    qlabel = signal_quality.get("quality_label", "unknown")
    if qlabel == "poor":
        parts.append("\n⚠️ Signal quality was POOR — measurements may be inaccurate.")
    elif qlabel == "acceptable":
        parts.append("\n📶 Signal quality was ACCEPTABLE.")

    parts.append(
        "\nPlease generate a brief, natural-language health summary in "
        f"{language}. Be specific about numbers, empathetic in tone, "
        "and include a disclaimer that this is not medical advice."
    )

    system = SYSTEM_PROMPT.format(language=language)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


# ── main generator class ────────────────────────────────────────────────────
class LLMHealthReporter:
    """LLM-powered health report generator."""

    def __init__(self, backend: Optional[LLMBackend] = None, language: str = "Chinese"):
        self.backend = backend or get_backend()
        self.language = language

    def generate_report(
        self,
        classifications: List[Dict],
        signal_quality: Dict,
        biomarkers: Optional[Dict] = None,
    ) -> str:
        """Generate a natural-language health report."""
        messages = _build_prompt(classifications, signal_quality, self.language)
        try:
            return self.backend.chat(messages, temperature=0.3, max_tokens=500)
        except Exception as e:
            # Fallback to template if LLM fails
            from src.nl_generation.nl_generator import generate_health_report
            return (
                f"[LLM unavailable: {e}]\n\n"
                + generate_health_report(classifications, signal_quality)
            )

    def generate_compact(
        self,
        classifications: List[Dict],
        signal_quality: Dict,
    ) -> str:
        """Generate a compact single-paragraph summary."""
        messages = _build_prompt(classifications, signal_quality, self.language)
        # Append a compactness instruction
        messages.append({
            "role": "user",
            "content": "Make this EXTREMELY concise — a single paragraph under 80 words."
        })
        try:
            return self.backend.chat(messages, temperature=0.2, max_tokens=200)
        except Exception as e:
            from src.nl_generation.nl_generator import generate_compact_nl
            return f"[LLM unavailable: {e}] {generate_compact_nl(classifications, signal_quality)}"


# ── quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with simulated data
    mock_classifications = [
        {
            "metric_name": "heart_rate",
            "value": 105.0, "unit": "bpm",
            "category": "tachycardia", "risk_level": "danger",
            "normal_range": "60-100 bpm",
        },
        {
            "metric_name": "systolic_bp",
            "value": 128.0, "unit": "mmHg",
            "category": "elevated", "risk_level": "caution",
            "normal_range": "90-120 mmHg",
        },
        {
            "metric_name": "spo2",
            "value": 97.0, "unit": "%",
            "category": "normal", "risk_level": "healthy",
            "normal_range": "95-100 %",
        },
    ]
    mock_quality = {"quality_label": "good", "snr_db": 15.0}

    reporter = LLMHealthReporter()
    report = reporter.generate_report(mock_classifications, mock_quality)
    print("=== LLM Health Report ===")
    print(report)
