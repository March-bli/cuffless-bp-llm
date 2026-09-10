"""
Stage 1: Signal Processing Layer
Extract physiological biomarkers from raw PPG/ECG signals using NeuroKit2.
"""

import numpy as np
import neurokit2 as nk
from typing import Dict


def extract_biomarkers(ppg_signal: np.ndarray, sampling_rate: int = 100) -> Dict[str, float]:
    """
    Process raw PPG signal and extract key physiological biomarkers.

    Args:
        ppg_signal: Raw PPG waveform (1D array)
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dict of biomarker name -> value
    """
    biomarkers = {}

    # Clean PPG signal
    ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)

    # Find PPG peaks (NeuroKit2 returns a bitmask DataFrame, convert to indices)
    try:
        ppg_peaks, info = nk.ppg_peaks(ppg_clean, sampling_rate=sampling_rate)
        peak_mask = ppg_peaks["PPG_Peaks"].values
        peak_indices = np.where(peak_mask == 1)[0]
        if len(peak_indices) == 0:
            peak_indices = _simple_peak_detect(ppg_clean, sampling_rate)
    except Exception:
        peak_indices = _simple_peak_detect(ppg_clean, sampling_rate)

    if len(peak_indices) < 2:
        return biomarkers  # Not enough data

    # Heart rate from peak intervals
    ibi_ms = np.diff(peak_indices) / sampling_rate * 1000  # Inter-beat intervals in ms
    heart_rate_bpm = 60000 / np.mean(ibi_ms)
    biomarkers["heart_rate"] = round(float(heart_rate_bpm), 1)

    # HRV: SDNN
    if len(ibi_ms) >= 2:
        sdnn = np.std(ibi_ms)
        biomarkers["hrv_sdnn"] = round(float(sdnn), 1)

    # SpO2 estimation from PPG AC/DC ratio (approximation for red+IR)
    # Standard pulse oximetry: SpO2 ≈ 110 - 25*R, where R = (AC_red/DC_red)/(AC_ir/DC_ir)
    # Without dual-wavelength, approximate from signal quality
    try:
        ac_component = np.std(ppg_clean)
        dc_component = np.mean(np.abs(ppg_clean)) + 1e-6
        perfusion_index = ac_component / dc_component * 100
        # Rough SpO2 bound based on PI (not clinically accurate, but demonstrates the pipeline)
        spo2_est = min(100.0, max(90.0, 95.0 + perfusion_index * 0.5))
        biomarkers["spo2"] = round(float(spo2_est), 1)
    except Exception:
        biomarkers["spo2"] = 98.0  # Default fallback

    return biomarkers


def extract_biomarkers_with_ecg(
    ecg_signal: np.ndarray,
    ppg_signal: np.ndarray,
    sampling_rate: int = 125,
) -> Dict[str, float]:
    """
    Process ECG + PPG signals together for BP estimation.
    Uses PTT (Pulse Transit Time) method for cuffless BP estimation.

    Args:
        ecg_signal: Raw ECG waveform
        ppg_signal: Raw PPG waveform
        sampling_rate: Sampling frequency in Hz

    Returns:
        Dict of biomarker name -> value including SBP and DBP
    """
    biomarkers = {}

    # Clean signals
    ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)

    # Get PPG-based biomarkers first
    ppg_biomarkers = extract_biomarkers(ppg_signal, sampling_rate)
    biomarkers.update(ppg_biomarkers)

    # ECG R-peak detection (NeuroKit2 returns bitmask DataFrame)
    try:
        ecg_peaks, ecg_info = nk.ecg_peaks(ecg_clean, sampling_rate=sampling_rate)
        r_peak_mask = ecg_peaks["ECG_R_Peaks"].values
        r_peaks = np.where(r_peak_mask == 1)[0]
    except Exception:
        return biomarkers

    # PPG peak detection (bitmask to indices)
    try:
        ppg_peaks_data, _ = nk.ppg_peaks(ppg_clean, sampling_rate=sampling_rate)
        ppg_peak_mask = ppg_peaks_data["PPG_Peaks"].values
        ppg_peaks = np.where(ppg_peak_mask == 1)[0]
        if len(ppg_peaks) == 0:
            ppg_peaks = _simple_peak_detect(ppg_clean, sampling_rate)
    except Exception:
        ppg_peaks = _simple_peak_detect(ppg_clean, sampling_rate)

    # Calculate PTT for BP estimation
    if len(r_peaks) < 3 or len(ppg_peaks) < 3:
        return biomarkers

    # Match nearest ECG R-peak to each PPG foot
    ptt_values = []
    for ppg_peak in ppg_peaks:
        # Find nearest preceding R-peak
        preceding_r = r_peaks[r_peaks < ppg_peak]
        if len(preceding_r) > 0:
            r_peak = preceding_r[-1]
            ptt_ms = (ppg_peak - r_peak) / sampling_rate * 1000
            if 100 < ptt_ms < 500:  # Physiologically plausible range
                ptt_values.append(ptt_ms)

    if len(ptt_values) >= 2:
        mean_ptt = np.mean(ptt_values)
        # Bramwell-Hill model: BP = a - b * ln(PTT)
        # Using literature-derived coefficients (approximate)
        sbp = 215.0 - 18.5 * np.log(mean_ptt)
        dbp = 120.0 - 12.0 * np.log(mean_ptt)
        biomarkers["systolic_bp"] = round(float(sbp), 1)
        biomarkers["diastolic_bp"] = round(float(dbp), 1)

    return biomarkers


def _simple_peak_detect(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Simple peak detection fallback using scipy."""
    from scipy.signal import find_peaks

    height = np.mean(signal) + 0.3 * np.std(signal)
    distance = int(sampling_rate * 0.4)  # Min 400ms between peaks
    peaks, _ = find_peaks(signal, height=height, distance=distance)
    return peaks


def assess_signal_quality(ppg_signal: np.ndarray, sampling_rate: int) -> dict:
    """
    Assess PPG signal quality before biomarker extraction.
    Returns quality metrics that can feed into the NL generation.
    """
    quality = {}

    # Signal-to-noise ratio
    try:
        ppg_clean = nk.ppg_clean(ppg_signal, sampling_rate=sampling_rate)
        residual = ppg_signal - ppg_clean
        snr = 10 * np.log10(np.var(ppg_clean) / (np.var(residual) + 1e-10))
        quality["snr_db"] = round(float(snr), 1)
    except Exception:
        quality["snr_db"] = 0.0

    # Percentage of signal that is non-zero (simple coverage)
    quality["coverage"] = round(float(np.mean(np.abs(ppg_signal) > 1e-6) * 100), 1)

    # Quality label
    if quality.get("snr_db", 0) > 10:
        quality["quality_label"] = "good"
    elif quality.get("snr_db", 0) > 5:
        quality["quality_label"] = "acceptable"
    else:
        quality["quality_label"] = "poor"

    return quality
