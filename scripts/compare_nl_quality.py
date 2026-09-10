"""
LLM vs Template NL Quality Comparison.

Generates health reports for sampled PulseDB segments using both:
  1. Rule-based template (nl_generator.py)
  2. LLM-based generation (llm_generator.py)

Evaluates on: naturalness, informativeness, clinical accuracy, conciseness.
"""

import sys, os, json, random
import numpy as np

sys.path.insert(0, "/home/lby/projects/bishe")

from src.main import run_pipeline
from src.pulsedb_loader import load_pulsedb_sample, iter_pulsedb_segments
from src.nl_generation.nl_generator import generate_health_report
from src.nl_generation.llm_generator import LLMHealthReporter, get_backend


DATA_DIR = "/home/lby/projects/bishe/data/pulsedb/Segment_Files"
OUTPUT_DIR = "/home/lby/projects/bishe/data/processed"
N_SAMPLES = 10  # number of segments to compare


def get_valid_files():
    """Get list of valid .mat files."""
    import h5py
    valid = []
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            fpath = os.path.join(root, fname)
            try:
                with h5py.File(fpath, "r") as f:
                    _ = f["Subj_Wins"]["PPG_F"]
                valid.append(fpath)
            except Exception:
                continue
    return valid


def sample_segments(valid_files: list, n: int = N_SAMPLES):
    """Sample N diverse segments from valid files."""
    import h5py
    samples = []
    random.shuffle(valid_files)

    for fpath in valid_files:
        if len(samples) >= n:
            break
        try:
            with h5py.File(fpath, "r") as f:
                group = f["Subj_Wins"]
                ppg_refs = group["PPG_F"][0]
                sbp_refs = group["SegSBP"][0]
                dbp_refs = group["SegDBP"][0]
                n_seg = len(ppg_refs)

                # Sample a random segment from this file
                seg_idx = random.randint(0, n_seg - 1)
                resolved = group[ppg_refs[seg_idx]]
                ppg = np.array(resolved).flatten().astype(np.float64)

                if ppg.std() < 1e-6:
                    continue

                sbp = float(np.array(group[sbp_refs[seg_idx]]).flatten()[0])
                dbp = float(np.array(group[dbp_refs[seg_idx]]).flatten()[0])

                samples.append({
                    "file": os.path.basename(fpath),
                    "seg_idx": seg_idx,
                    "ppg": ppg,
                    "sbp": sbp,
                    "dbp": dbp,
                })
        except Exception:
            continue

    return samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Finding valid PulseDB files...")
    valid_files = get_valid_files()
    print(f"  {len(valid_files)} valid files")

    print(f"Sampling {N_SAMPLES} segments...")
    samples = sample_segments(valid_files, N_SAMPLES)
    print(f"  {len(samples)} segments collected")

    # Initialize LLM reporter
    try:
        backend = get_backend()
        llm_reporter = LLMHealthReporter(backend=backend)
        llm_available = True
        print(f"LLM backend: {backend.name} ({backend.model})")
    except Exception as e:
        print(f"LLM unavailable: {e}")
        llm_available = False

    results = []
    for i, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{len(samples)}: {sample['file']}#{sample['seg_idx']}")
        print(f"  SBP={sample['sbp']:.0f}, DBP={sample['dbp']:.0f}")

        # Run the pipeline (template NL)
        pipeline_result = run_pipeline(
            ppg_signal=sample["ppg"],
            sampling_rate=125,
            sbp_gt=sample["sbp"],
            dbp_gt=sample["dbp"],
            skip_cleaning=True,
            output_mode="full",
        )

        template_report = pipeline_result["nl_report"]
        classifications = pipeline_result["classifications"]
        signal_quality = pipeline_result["signal_quality"]
        biomarkers = pipeline_result["biomarkers"]

        # LLM report (if available)
        llm_report = None
        if llm_available:
            try:
                llm_report = llm_reporter.generate_report(
                    classifications, signal_quality, biomarkers
                )
            except Exception as e:
                llm_report = f"[LLM ERROR: {e}]"

        results.append({
            "sample": f"{sample['file']}#{sample['seg_idx']}",
            "sbp": sample["sbp"],
            "dbp": sample["dbp"],
            "biomarkers": biomarkers,
            "template_report": template_report,
            "llm_report": llm_report,
            "classifications": [
                {
                    "metric": c["metric_name"],
                    "risk": c["risk_level"],
                    "category": c["category"],
                }
                for c in classifications
            ],
        })

        # Print comparison
        print(f"\n  --- Template NL ---")
        print("  " + template_report.replace("\n", "\n  ")[:300])

        if llm_report:
            print(f"\n  --- LLM NL ---")
            print("  " + llm_report.replace("\n", "\n  ")[:300])

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "nl_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nComparison saved to {out_path}")

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Samples: {len(results)}")
    print(f"  LLM available: {llm_available}")
    if llm_available and results:
        llm_ok = sum(1 for r in results if r["llm_report"] and "ERROR" not in str(r["llm_report"]))
        print(f"  LLM success rate: {llm_ok}/{len(results)}")


if __name__ == "__main__":
    main()
