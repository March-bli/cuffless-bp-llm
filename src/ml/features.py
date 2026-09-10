"""
PPG Feature Extraction for Blood Pressure Estimation.

Extracts morphological, derivative (VPG/APG), statistical, and frequency-domain
features from cleaned PPG waveforms. Based on features used in cuffless BP literature
(Elgendi 2012, Liang 2018, Slapnicar 2019).
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.signal import find_peaks
from typing import Dict, List, Optional, Tuple


# ── peak detection ────────────────────────────────────────────────────────
def _find_ppg_peaks(ppg: np.ndarray, fs: int) -> Tuple[np.ndarray, dict]:
    """Find systolic peaks in a PPG segment."""
    height = np.median(ppg) + 0.2 * (np.max(ppg) - np.median(ppg))
    distance = int(fs * 0.4)  # min 400 ms between beats (~150 bpm max)
    peaks, props = find_peaks(ppg, height=height, distance=distance)
    return peaks, props


def _find_ppg_onsets(ppg: np.ndarray, peaks: np.ndarray, fs: int) -> np.ndarray:
    """Find PPG pulse onset (foot) before each systolic peak."""
    onsets = []
    search_dist = int(fs * 0.3)  # look back up to 300 ms
    for p in peaks:
        start = max(0, p - search_dist)
        if start < p:
            onset = start + np.argmin(ppg[start:p])
            onsets.append(onset)
    return np.array(onsets, dtype=int)


# ── time-domain features ──────────────────────────────────────────────────
def _time_features(ppg: np.ndarray, peaks: np.ndarray,
                   onsets: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract time-domain morphological features."""
    f = {}
    if len(peaks) < 2 or len(onsets) < 2:
        return f

    # --- amplitudes ---
    peak_amps = ppg[peaks]
    onset_amps = ppg[onsets]
    f["sys_amp_mean"] = float(np.mean(peak_amps))
    f["sys_amp_std"] = float(np.std(peak_amps))
    f["dia_amp_mean"] = float(np.mean(onset_amps))
    f["pulse_amp_mean"] = f["sys_amp_mean"] - f["dia_amp_mean"]
    f["pulse_amp_std"] = float(np.std(peak_amps - onset_amps[:len(peaks)]))

    # --- timing ---
    pulse_intervals = np.diff(peaks) / fs * 1000  # ms
    f["pulse_interval_mean"] = float(np.mean(pulse_intervals))
    f["pulse_interval_std"] = float(np.std(pulse_intervals))
    f["heart_rate"] = 60000.0 / f["pulse_interval_mean"]

    # systolic upstroke time (onset → peak)
    valid_pairs = min(len(peaks), len(onsets))
    upstroke_times = (peaks[:valid_pairs] - onsets[:valid_pairs]) / fs * 1000
    f["upstroke_time_mean"] = float(np.mean(upstroke_times))
    f["upstroke_time_std"] = float(np.std(upstroke_times))

    # diastolic time (peak → next onset)
    if len(peaks) >= 2 and len(onsets) >= 2:
        dia_times = (onsets[1:valid_pairs] - peaks[:valid_pairs-1]) / fs * 1000
        if len(dia_times) > 0:
            f["diastolic_time_mean"] = float(np.mean(dia_times))

    # --- pulse width at different heights ---
    for pct in [25, 50, 75]:
        widths = []
        for p in peaks[:min(10, len(peaks))]:
            thresh = onset_amps[min(len(onsets)-1, list(peaks).index(p))] + \
                     (ppg[p] - onset_amps[min(len(onsets)-1, list(peaks).index(p))]) * pct / 100
            if np.isnan(thresh):
                continue
            left = p
            while left > 0 and ppg[left] > thresh:
                left -= 1
            right = p
            while right < len(ppg) - 1 and ppg[right] > thresh:
                right += 1
            widths.append((right - left) / fs * 1000)
        if widths:
            f[f"pulse_width_{pct}pct"] = float(np.mean(widths))

    # --- augmentation index (AIx) ---
    # AIx = (P2 - P1) / PP * 100 where P2=reflected wave, P1=systolic peak
    # Approximate: ratio of late systolic inflection to early systolic peak
    if len(peaks) >= 2:
        a = ppg[peaks[1:]] - ppg[onsets[1:len(peaks)]]
        b = ppg[peaks[:-1]] - ppg[onsets[:len(peaks)-1]]
        if len(a) > 0 and len(b) > 0 and np.mean(b) > 0:
            f["aix"] = float(np.mean(a) / np.mean(b) * 100)

    # --- large artery stiffness index (SI) ---
    # SI = height / (systolic peak - diastolic peak) time
    if "pulse_interval_mean" in f and "upstroke_time_mean" in f:
        f["stiffness_index"] = float(
            f["pulse_interval_mean"] / (f["upstroke_time_mean"] + 1e-6)
        )

    return f


# ── derivative features (VPG / APG) ────────────────────────────────────────
def _derivative_features(ppg: np.ndarray, peaks: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract first-derivative (VPG) and second-derivative (APG) features.

    APG waveform landmarks (a, b, c, d, e waves) per Elgendi 2012.
    """
    f = {}
    vpg = np.gradient(ppg)       # first derivative (velocity)
    apg = np.gradient(vpg)        # second derivative (acceleration)

    # VPG features
    f["vpg_max"] = float(np.max(vpg))
    f["vpg_min"] = float(np.min(vpg))
    f["vpg_std"] = float(np.std(vpg))
    f["vpg_mean_abs"] = float(np.mean(np.abs(vpg)))

    # APG features (a-b-c-d-e waves) per cardiac cycle
    if len(peaks) < 2:
        return f

    a_waves, b_waves, c_waves, d_waves, e_waves = [], [], [], [], []
    search_range = int(fs * 0.35)  # 350 ms around each beat

    for p in peaks:
        start = max(0, p - search_range)
        end = min(len(apg), p + search_range)
        beat_apg = apg[start:end]

        if len(beat_apg) < 10:
            continue

        # Find local extrema in the APG within this beat
        apg_peaks_idx, _ = find_peaks(beat_apg)
        apg_valleys_idx, _ = find_peaks(-beat_apg)

        apg_peaks_vals = beat_apg[apg_peaks_idx] if len(apg_peaks_idx) > 0 else np.array([])
        apg_valleys_vals = beat_apg[apg_valleys_idx] if len(apg_valleys_idx) > 0 else np.array([])

        # Sort peaks by amplitude (descending) → a, c, e waves
        sorted_peak_idx = np.argsort(apg_peaks_vals)[::-1]
        # Sort valleys by amplitude (ascending, most negative first) → b, d waves
        sorted_valley_idx = np.argsort(apg_valleys_vals)

        if len(sorted_peak_idx) >= 1:
            a_waves.append(apg_peaks_vals[sorted_peak_idx[0]])
        if len(sorted_valley_idx) >= 1:
            b_waves.append(apg_valleys_vals[sorted_valley_idx[0]])
        if len(sorted_peak_idx) >= 2:
            c_waves.append(apg_peaks_vals[sorted_peak_idx[1]])
        if len(sorted_valley_idx) >= 2:
            d_waves.append(apg_valleys_vals[sorted_valley_idx[1]])
        if len(sorted_peak_idx) >= 3:
            e_waves.append(apg_peaks_vals[sorted_peak_idx[2]])

    for name, waves in [("a", a_waves), ("b", b_waves), ("c", c_waves),
                         ("d", d_waves), ("e", e_waves)]:
        if waves:
            f[f"apg_{name}_mean"] = float(np.mean(waves))
            f[f"apg_{name}_std"] = float(np.std(waves))

    # APG ratios (b/a, c/a, d/a, e/a) — important for BP estimation
    if a_waves:
        a_mean = np.mean(a_waves)
        for name, waves in [("b", b_waves), ("c", c_waves),
                             ("d", d_waves), ("e", e_waves)]:
            if waves and a_mean != 0:
                f[f"apg_{name}_a_ratio"] = float(np.mean(waves) / a_mean)

    return f


# ── statistical features ──────────────────────────────────────────────────
def _statistical_features(ppg: np.ndarray) -> Dict[str, float]:
    """Extract global statistical features."""
    from scipy.stats import skew, kurtosis
    f = {}
    f["ppg_mean"] = float(np.mean(ppg))
    f["ppg_std"] = float(np.std(ppg))
    f["ppg_median"] = float(np.median(ppg))
    f["ppg_skew"] = float(skew(ppg))
    f["ppg_kurtosis"] = float(kurtosis(ppg))
    f["ppg_min"] = float(np.min(ppg))
    f["ppg_max"] = float(np.max(ppg))
    f["ppg_ptp"] = float(np.ptp(ppg))  # peak-to-peak amplitude
    f["ppg_rms"] = float(np.sqrt(np.mean(ppg ** 2)))

    # Percentiles
    for pct in [5, 25, 75, 95]:
        f[f"ppg_p{pct}"] = float(np.percentile(ppg, pct))

    # Entropy (approximate)
    hist, _ = np.histogram(ppg, bins=20, density=True)
    hist = hist[hist > 0]
    f["ppg_entropy"] = float(-np.sum(hist * np.log2(hist)))

    return f


# ── frequency-domain features ─────────────────────────────────────────────
def _frequency_features(ppg: np.ndarray, fs: int) -> Dict[str, float]:
    """Extract frequency-domain features via FFT."""
    f = {}
    n = len(ppg)
    fft = np.abs(np.fft.rfft(ppg))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)

    # Total spectral power
    total_power = np.sum(fft ** 2)
    f["spectral_energy"] = float(total_power)

    # Power in heart-rate band (0.5–5 Hz)
    hr_band = (freqs >= 0.5) & (freqs <= 5.0)
    hr_power = np.sum(fft[hr_band] ** 2)
    f["hr_band_power"] = float(hr_power)
    f["hr_band_ratio"] = float(hr_power / (total_power + 1e-12))

    # Dominant frequency
    if total_power > 0:
        dominant_idx = np.argmax(fft[1:]) + 1  # skip DC
        f["dominant_freq"] = float(freqs[dominant_idx])
        f["dominant_power"] = float(fft[dominant_idx] ** 2)

    # Spectral centroid
    if total_power > 0:
        f["spectral_centroid"] = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-12))

    # Low-frequency (0–0.5 Hz) vs high-frequency (0.5–5 Hz) ratio
    lf_band = freqs < 0.5
    lf_power = np.sum(fft[lf_band] ** 2)
    f["lf_hf_ratio"] = float(lf_power / (hr_power + 1e-12))

    return f


# ── main extraction entry ─────────────────────────────────────────────────
def extract_ppg_features(
    ppg: np.ndarray,
    sampling_rate: int = 125,
    include_derivative: bool = True,
    include_frequency: bool = True,
) -> Dict[str, float]:
    """
    Extract comprehensive PPG features for blood pressure estimation.

    Args:
        ppg: Cleaned PPG waveform (1-D array)
        sampling_rate: Sampling frequency in Hz
        include_derivative: Include VPG/APG derivative features
        include_frequency: Include FFT frequency features

    Returns:
        Flat dict of feature_name → value (typically 40–60 features)
    """
    features: Dict[str, float] = {}

    # Peak detection
    peaks, _ = _find_ppg_peaks(ppg, sampling_rate)
    onsets = _find_ppg_onsets(ppg, peaks, sampling_rate)

    if len(peaks) < 2:
        # Fallback: only statistical features when peaks are unreliable
        features.update(_statistical_features(ppg))
        return features

    # Time-domain
    features.update(_time_features(ppg, peaks, onsets, sampling_rate))

    # Derivative (VPG + APG)
    if include_derivative:
        features.update(_derivative_features(ppg, peaks, sampling_rate))

    # Statistical
    features.update(_statistical_features(ppg))

    # Frequency
    if include_frequency:
        features.update(_frequency_features(ppg, sampling_rate))

    # Remove NaN/Inf values
    features = {k: v for k, v in features.items()
                if np.isfinite(v) and not np.isnan(v)}

    return features


def extract_features_batch(
    ppg_segments: List[np.ndarray],
    sampling_rate: int = 125,
) -> np.ndarray:
    """
    Extract features from a batch of PPG segments.
    Returns feature matrix [n_segments, n_features].
    """
    all_features = []
    for ppg in ppg_segments:
        feats = extract_ppg_features(ppg, sampling_rate)
        all_features.append(feats)

    # Build consistent feature matrix
    # Union all feature names
    all_keys = set()
    for f in all_features:
        all_keys.update(f.keys())
    all_keys = sorted(all_keys)

    X = np.zeros((len(all_features), len(all_keys)))
    for i, feats in enumerate(all_features):
        for j, key in enumerate(all_keys):
            X[i, j] = feats.get(key, 0.0)

    return X, all_keys
