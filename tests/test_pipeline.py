"""
Test the full pipeline with simulated PPG/ECG data.
"""

import sys
sys.path.insert(0, "/home/lby/projects/bishe")

import numpy as np
import neurokit2 as nk
from src.main import run_pipeline


def simulate_healthy_ppg(duration_sec: int = 10, sampling_rate: int = 100) -> np.ndarray:
    """Simulate a healthy PPG signal at ~72 bpm."""
    ppg = nk.ppg_simulate(
        duration=duration_sec,
        sampling_rate=sampling_rate,
        heart_rate=72,
        frequency_modulation=0.05,
    )
    return ppg


def simulate_elevated_hr_ppg(duration_sec: int = 10, sampling_rate: int = 100) -> np.ndarray:
    """Simulate PPG with elevated heart rate (~105 bpm, tachycardia)."""
    ppg = nk.ppg_simulate(
        duration=duration_sec,
        sampling_rate=sampling_rate,
        heart_rate=105,
        frequency_modulation=0.03,
    )
    return ppg


def simulate_bradycardia_ppg(duration_sec: int = 10, sampling_rate: int = 100) -> np.ndarray:
    """Simulate PPG with low heart rate (~50 bpm, bradycardia)."""
    ppg = nk.ppg_simulate(
        duration=duration_sec,
        sampling_rate=sampling_rate,
        heart_rate=50,
        frequency_modulation=0.02,
    )
    return ppg


def simulate_ecg(duration_sec: int = 10, sampling_rate: int = 125, heart_rate: int = 72) -> np.ndarray:
    """Simulate ECG signal."""
    ecg = nk.ecg_simulate(
        duration=duration_sec,
        sampling_rate=sampling_rate,
        heart_rate=heart_rate,
    )
    return ecg


def test_healthy_scenario():
    """Test: healthy person (~72 bpm, normal vitals)."""
    print("\n" + "=" * 70)
    print("SCENARIO 1: Healthy Adult (HR ~72 bpm)")
    print("=" * 70)

    ppg = simulate_healthy_ppg(duration_sec=30, sampling_rate=128)
    ecg = simulate_ecg(duration_sec=30, sampling_rate=128, heart_rate=72)

    result = run_pipeline(
        ppg_signal=ppg,
        sampling_rate=128,
        ecg_signal=ecg,
        output_mode="full",
    )

    print(result["nl_report"])
    return result


def test_tachycardia_scenario():
    """Test: elevated heart rate (~105 bpm)."""
    print("\n" + "=" * 70)
    print("SCENARIO 2: Elevated Heart Rate (~105 bpm)")
    print("=" * 70)

    ppg = simulate_elevated_hr_ppg(duration_sec=30, sampling_rate=128)
    ecg = simulate_ecg(duration_sec=30, sampling_rate=128, heart_rate=105)

    result = run_pipeline(
        ppg_signal=ppg,
        sampling_rate=128,
        ecg_signal=ecg,
        output_mode="full",
    )

    print(result["nl_report"])
    return result


def test_bradycardia_scenario():
    """Test: low heart rate (~50 bpm)."""
    print("\n" + "=" * 70)
    print("SCENARIO 3: Bradycardia (~50 bpm)")
    print("=" * 70)

    ppg = simulate_bradycardia_ppg(duration_sec=30, sampling_rate=128)
    ecg = simulate_ecg(duration_sec=30, sampling_rate=128, heart_rate=50)

    result = run_pipeline(
        ppg_signal=ppg,
        sampling_rate=128,
        ecg_signal=ecg,
        output_mode="full",
    )

    print(result["nl_report"])
    return result


def test_compact_output():
    """Test compact NL output mode."""
    print("\n" + "=" * 70)
    print("SCENARIO 4: Compact NL Output (for AI assistant)")
    print("=" * 70)

    ppg = simulate_healthy_ppg(duration_sec=10, sampling_rate=128)
    result = run_pipeline(
        ppg_signal=ppg,
        sampling_rate=128,
        output_mode="compact",
    )

    print(result["nl_report"])
    print("\nBiomarkers:", result["biomarkers"])
    print("Clinical summary:", result["clinical_summary"])
    return result


if __name__ == "__main__":
    print("Testing BioHealth NL Pipeline")
    print("=" * 70)

    test_healthy_scenario()
    test_tachycardia_scenario()
    test_bradycardia_scenario()
    test_compact_output()

    print("\n" + "=" * 70)
    print("All pipeline tests completed.")
