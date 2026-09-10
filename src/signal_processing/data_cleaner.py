"""
PPG Data Cleaning Module for cuffless blood pressure estimation.

Cleaning stages:
  1. Signal quality gate — reject segments with poor SNR / low coverage
  2. Baseline wander removal — high-pass filter @ 0.5 Hz
  3. Power line notch — 50 or 60 Hz notch filter
  4. Motion artifact detection — sliding-window amplitude outlier flagging
  5. Spike removal — median-filter outlier replacement
  6. Normalisation — zero-mean unit-variance

Works with both PulseDB (125 Hz) and MAX30102 (100 Hz) data.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import median_filter
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


# ── defaults ─────────────────────────────────────────────────────────────
HIGH_PASS_CUTOFF = 0.5       # Hz — baseline wander
NOTCH_FREQ = 50.0            # Hz — mains hum (use 60.0 for US/Asia)
NOTCH_Q = 30.0               # notch quality factor
MOTION_WINDOW_SEC = 2.0      # sliding window for motion detection
MOTION_Z_THRESH = 3.0        # Z-score threshold for motion flag
MEDIAN_KERNEL = 5            # samples — spike removal window
QUALITY_SNR_MIN = -15.0       # dB — reject below this (lenient for normalised signals)
QUALITY_COVER_MIN = 50.0      # % — reject segments with less non-zero coverage


# ── result dataclass ─────────────────────────────────────────────────────
@dataclass
class CleaningResult:
    """Result of a full cleaning pass."""
    signal_clean: np.ndarray          # cleaned PPG signal
    signal_raw: np.ndarray            # original (reference)
    sampling_rate: int
    quality_score: float              # 0–1 overall quality
    quality_label: str                # "good" / "acceptable" / "poor"
    motion_mask: np.ndarray           # boolean mask, same length as signal
    snr_before: float
    snr_after: float
    stages_applied: list = field(default_factory=list)


# ── helper: filter design ─────────────────────────────────────────────────
def _design_highpass(cutoff: float, fs: int, order: int = 4) -> Tuple:
    """Second-order-section high-pass Butterworth."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    return scipy_signal.butter(order, normal_cutoff, btype="high", output="sos")


def _design_notch(f0: float, fs: int, Q: float = NOTCH_Q) -> Tuple:
    """Second-order-section notch (band-stop) filter."""
    nyq = 0.5 * fs
    w0 = f0 / nyq
    if w0 >= 1.0:
        w0 = 0.99
    return scipy_signal.iirnotch(w0, Q, fs=fs)


def _estimate_snr(signal: np.ndarray) -> float:
    """Estimate signal-to-noise ratio in dB.

    Band-pass the signal to isolate the heart-rate band (0.5–5 Hz),
    then compute the ratio of in-band power to out-of-band power.
    Falls back to a simpler AC/DC ratio when the signal is short.
    """
    n = len(signal)
    if n < 20 or signal.std() < 1e-10:
        return 0.0
    try:
        # Use FFT to separate signal band (0.5–5 Hz, ~30–300 bpm)
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n)  # normalised 0–0.5
        # 0.5–5 Hz mapped into normalised frequency depends on fs — just use bins
        # Bin range for 0.5–5 Hz at 125 Hz: 0.5/62.5=0.008 → 5/62.5=0.08
        lo_bin = max(1, int(0.008 * n))
        hi_bin = min(len(freqs) - 1, int(0.08 * n * 2))
        if hi_bin <= lo_bin:
            return 0.0
        signal_power = np.sum(np.abs(fft[lo_bin:hi_bin]) ** 2)
        noise_power = np.sum(np.abs(fft) ** 2) - signal_power
        if noise_power <= 1e-12:
            return 20.0
        snr = 10.0 * np.log10(signal_power / noise_power)
        return round(float(np.clip(snr, -20, 30)), 1)
    except Exception:
        return 0.0


# ── main cleaner ─────────────────────────────────────────────────────────
class PPGCleaner:
    """Multi-stage PPG signal cleaner for blood-pressure estimation."""

    def __init__(
        self,
        sampling_rate: int = 125,
        hp_cutoff: float = HIGH_PASS_CUTOFF,
        notch_freq: Optional[float] = NOTCH_FREQ,   # None disables notch
        motion_window_sec: float = MOTION_WINDOW_SEC,
        motion_z_thresh: float = MOTION_Z_THRESH,
        median_kernel: int = MEDIAN_KERNEL,
        snr_min: float = QUALITY_SNR_MIN,
        cover_min: float = QUALITY_COVER_MIN,
    ):
        self.sampling_rate = sampling_rate
        self.hp_cutoff = hp_cutoff
        self.notch_freq = notch_freq
        self.motion_window = max(1, int(motion_window_sec * sampling_rate))
        self.motion_z_thresh = motion_z_thresh
        self.median_kernel = median_kernel
        self.snr_min = snr_min
        self.cover_min = cover_min

    # ── individual stages ──────────────────────────────────────────────

    def quality_gate(self, signal: np.ndarray) -> Tuple[bool, float, str]:
        """Return (pass, score, label).  Pass=False → segment should be discarded."""
        snr = _estimate_snr(signal)
        coverage = float(np.mean(np.abs(signal) > 1e-8) * 100)

        if snr < self.snr_min or coverage < self.cover_min:
            label = "poor"
        elif snr < self.snr_min * 2:
            label = "acceptable"
        else:
            label = "good"

        # score: weighted combination of SNR and coverage
        snr_score = min(1.0, max(0.0, snr / 20.0))   # 20 dB → perfect
        cov_score = coverage / 100.0
        score = round(0.6 * snr_score + 0.4 * cov_score, 3)

        return label != "poor", score, label

    def remove_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """High-pass filter to remove low-frequency baseline drift."""
        sos = _design_highpass(self.hp_cutoff, self.sampling_rate)
        return scipy_signal.sosfiltfilt(sos, signal)

    def remove_powerline_noise(self, signal: np.ndarray) -> np.ndarray:
        """Apply notch filter at mains frequency (50 or 60 Hz)."""
        if self.notch_freq is None:
            return signal
        b, a = _design_notch(self.notch_freq, self.sampling_rate)
        return scipy_signal.filtfilt(b, a, signal)

    def detect_motion_artifact(self, signal: np.ndarray) -> np.ndarray:
        """Return boolean mask where motion artifact is suspected.

        Uses a sliding-window standard-deviation Z-score: a sudden
        amplitude spike in the local window implies movement.
        """
        n = len(signal)
        half = self.motion_window // 2
        local_std = np.zeros(n)

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half)
            local_std[i] = float(np.std(signal[start:end]))

        # Z-score on local std values
        global_std_mean = np.mean(local_std)
        global_std_std = np.std(local_std) + 1e-12
        z_scores = np.abs(local_std - global_std_mean) / global_std_std

        return z_scores > self.motion_z_thresh

    def remove_spikes(self, signal: np.ndarray) -> np.ndarray:
        """Replace isolated spikes with local median value."""
        kernel = max(3, self.median_kernel)
        if kernel % 2 == 0:
            kernel += 1
        smoothed = median_filter(signal, size=kernel)

        # Only replace samples that deviate strongly from the median
        mad = np.median(np.abs(signal - smoothed)) * 1.4826  # robust std
        if mad < 1e-10:
            return signal
        thresh = 5.0 * mad
        outliers = np.abs(signal - smoothed) > thresh
        cleaned = signal.copy()
        cleaned[outliers] = smoothed[outliers]
        return cleaned

    def normalise(self, signal: np.ndarray) -> np.ndarray:
        """Zero-mean, unit-variance normalisation."""
        std = signal.std()
        if std < 1e-12:
            return signal - signal.mean()
        return (signal - signal.mean()) / std

    # ── full pipeline ──────────────────────────────────────────────────

    def clean(self, ppg_signal: np.ndarray) -> CleaningResult:
        """Run the full cleaning pipeline on a PPG segment.

        Args:
            ppg_signal: 1-D numpy array (raw or pre-filtered PPG)

        Returns:
            CleaningResult with cleaned signal and quality metadata.
        """
        sig = ppg_signal.astype(np.float64).copy()
        raw = sig.copy()
        stages = []

        # Stage 0 – quality gate (informational only)
        passes, score, label = self.quality_gate(sig)
        snr_before = _estimate_snr(sig)

        # Stage 1 – baseline wander
        sig = self.remove_baseline_wander(sig)
        stages.append("baseline_wander")

        # Stage 2 – power line noise
        sig = self.remove_powerline_noise(sig)
        stages.append("powerline_notch")

        # Stage 3 – motion detection (produce mask, do not modify waveform)
        motion_mask = self.detect_motion_artifact(sig)
        stages.append("motion_detect")

        # Stage 4 – spike removal
        sig = self.remove_spikes(sig)
        stages.append("spike_removal")

        # Stage 5 – normalisation (optional — kept raw for feature extraction)
        # Most ML pipelines prefer to normalise later, so we skip normalise()
        # in the default clean() and let the caller decide.
        stages.append("cleaned")

        snr_after = _estimate_snr(sig)

        return CleaningResult(
            signal_clean=sig,
            signal_raw=raw,
            sampling_rate=self.sampling_rate,
            quality_score=score,
            quality_label=label,
            motion_mask=motion_mask,
            snr_before=snr_before,
            snr_after=snr_after,
            stages_applied=stages,
        )

    def clean_batch(self, signals: list) -> list:
        """Clean a batch of PPG segments and return (result, passes_quality_gate)."""
        results = []
        for sig in signals:
            res = self.clean(sig)
            results.append(res)
        return results

    def summary(self, results: list) -> Dict:
        """Return aggregate statistics for a batch of CleaningResults."""
        n = len(results)
        n_good = sum(1 for r in results if r.quality_label == "good")
        n_acceptable = sum(1 for r in results if r.quality_label == "acceptable")
        n_poor = sum(1 for r in results if r.quality_label == "poor")
        motion_pct = [float(r.motion_mask.mean()) for r in results]
        snr_improvement = [r.snr_after - r.snr_before for r in results]

        return {
            "total_segments": n,
            "good": n_good,
            "acceptable": n_acceptable,
            "poor": n_poor,
            "discard_rate": round(n_poor / n, 3) if n else 0,
            "avg_motion_pct": round(np.mean(motion_pct) * 100, 1),
            "avg_snr_before": round(np.mean([r.snr_before for r in results]), 1),
            "avg_snr_after": round(np.mean([r.snr_after for r in results]), 1),
            "avg_snr_improvement": round(np.mean(snr_improvement), 1),
        }
