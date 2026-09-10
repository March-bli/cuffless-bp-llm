"""
Main pipeline: PulseDB raw signals -> NL health report.
Orchestrates Cleaning -> Stage 1 -> Stage 2 -> Stage 3.
"""

import numpy as np
from typing import Dict, Optional

from src.signal_processing.data_cleaner import PPGCleaner, CleaningResult
from src.signal_processing.signal_processor import (
    extract_biomarkers,
    extract_biomarkers_with_ecg,
    assess_signal_quality,
)
from src.clinical_knowledge.clinical_classifier import (
    classify_all_biomarkers,
    generate_clinical_summary,
)
from src.nl_generation.nl_generator import generate_health_report, generate_compact_nl
from src.nl_generation.llm_generator import LLMHealthReporter, get_backend
from src.utils.config import CLINICAL_RANGES


def run_pipeline(
    ppg_signal: np.ndarray,
    sampling_rate: int = 100,
    ecg_signal: Optional[np.ndarray] = None,
    duration_sec: Optional[int] = None,
    output_mode: str = "full",
    sbp_gt: Optional[float] = None,
    dbp_gt: Optional[float] = None,
    skip_cleaning: bool = False,
    use_llm: bool = False,
) -> Dict:
    """
    Run the full end-to-end pipeline.

    Args:
        ppg_signal: Raw PPG waveform
        sampling_rate: Sampling frequency in Hz
        ecg_signal: Optional raw ECG waveform (enables BP estimation via PTT)
        duration_sec: Measurement duration. Auto-calculated if not given.
        output_mode: "full" for health report, "compact" for chat/vocal output.
        sbp_gt: Optional ground-truth systolic BP (from PulseDB labels).
        dbp_gt: Optional ground-truth diastolic BP (from PulseDB labels).
        skip_cleaning: If True, skip the data cleaning stage.
        use_llm: If True, use LLM API for NL generation instead of template.

    Returns:
        {
            "biomarkers": {...},
            "classifications": [...],
            "signal_quality": {...},
            "cleaning_result": CleaningResult,  # None if skip_cleaning=True
            "clinical_summary": "...",
            "nl_report": "..."
        }
    """

    cleaning_result = None

    # Stage 0: Data Cleaning
    if not skip_cleaning:
        cleaner = PPGCleaner(sampling_rate=sampling_rate)
        cleaning_result = cleaner.clean(ppg_signal)
        ppg_signal = cleaning_result.signal_clean
        # Log quality warning for downstream stages
        if cleaning_result.quality_label == "poor":
            cleaning_result = cleaning_result  # keep for reference

    # Stage 1: Signal Processing
    if ecg_signal is not None and len(ecg_signal) > 0:
        biomarkers = extract_biomarkers_with_ecg(ecg_signal, ppg_signal, sampling_rate)
    else:
        biomarkers = extract_biomarkers(ppg_signal, sampling_rate)

    # Override BP estimates with ground-truth labels when available (PulseDB)
    if sbp_gt is not None:
        biomarkers["systolic_bp"] = round(float(sbp_gt), 1)
    if dbp_gt is not None:
        biomarkers["diastolic_bp"] = round(float(dbp_gt), 1)

    signal_quality = assess_signal_quality(ppg_signal, sampling_rate)

    if duration_sec is None:
        duration_sec = len(ppg_signal) // sampling_rate

    # Stage 2: Clinical Knowledge Classification
    classifications = classify_all_biomarkers(biomarkers, CLINICAL_RANGES)
    clinical_summary = generate_clinical_summary(classifications)

    # Stage 3: NL Generation
    if use_llm:
        try:
            backend = get_backend()
            reporter = LLMHealthReporter(backend=backend)
            if output_mode == "compact":
                nl_report = reporter.generate_compact(classifications, signal_quality)
            else:
                nl_report = reporter.generate_report(
                    classifications, signal_quality, biomarkers
                )
        except Exception as e:
            # Fallback to template
            if output_mode == "compact":
                nl_report = generate_compact_nl(classifications, signal_quality)
            else:
                nl_report = generate_health_report(classifications, signal_quality, duration_sec)
            nl_report = f"[LLM fallback: {e}]\n\n{nl_report}"
    else:
        if output_mode == "compact":
            nl_report = generate_compact_nl(classifications, signal_quality)
        else:
            nl_report = generate_health_report(classifications, signal_quality, duration_sec)

    return {
        "biomarkers": biomarkers,
        "classifications": classifications,
        "signal_quality": signal_quality,
        "cleaning_result": cleaning_result,
        "clinical_summary": clinical_summary,
        "nl_report": nl_report,
    }
