"""
Batch feature extraction from PulseDB .mat files.
Walks through all valid files, extracts PPG features + BP labels,
and saves feature matrix for ML training.
"""

import os, sys, glob, time, json
import h5py
import numpy as np

sys.path.insert(0, "/home/lby/projects/bishe")

from src.ml.features import extract_ppg_features

# ── config ─────────────────────────────────────────────────────────────────
DATA_DIR = "/mnt/parscratch/users/acp25bl/pulsedb/Segment_Files"
OUTPUT_DIR = "/mnt/parscratch/users/acp25bl/processed"
SAMPLING_RATE = 125  # PulseDB standard
MIN_PPG_STD = 1e-6    # skip segments with zero variance
MAX_SEGMENTS = None   # set to a number for quick test, None = all


def is_valid_hdf5(filepath: str) -> bool:
    """Check if an .mat file can be opened by h5py."""
    try:
        with h5py.File(filepath, "r") as f:
            group = f.get("Subj_Wins", f)
            _ = group["PPG_F"]
        return True
    except Exception:
        return False


def load_segment(group, refs, idx: int, key: str) -> np.ndarray:
    """Load one segment from HDF5 references."""
    try:
        resolved = group[refs[idx]]
        return np.array(resolved).flatten().astype(np.float64)
    except Exception:
        return np.array([])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all .mat files
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if fname.endswith(".mat"):
                all_files.append(os.path.join(root, fname))

    print(f"Found {len(all_files)} .mat files total")

    # Filter to valid files only
    valid_files = [fp for fp in all_files if is_valid_hdf5(fp)]
    print(f"Valid (non-corrupted): {len(valid_files)}")

    # Collect features
    X_list = []        # feature vectors
    y_sbp_list = []    # systolic BP
    y_dbp_list = []    # diastolic BP
    segment_ids = []   # for traceability
    ages_list = []     # demographic: age
    genders_list = []  # demographic: gender
    subject_ids_list = []  # subject ID for grouping
    skipped = 0
    flat_skipped = 0

    t0 = time.time()
    for fi, fpath in enumerate(valid_files):
        fname = os.path.basename(fpath)
        with h5py.File(fpath, "r") as f:
            group = f["Subj_Wins"]
            ppg_refs = group["PPG_F"][0]
            sbp_refs = group["SegSBP"][0]
            dbp_refs = group["SegDBP"][0]
            age_refs = group["Age"][0] if "Age" in group else None
            gender_refs = group["Gender"][0] if "Gender" in group else None
            subj_refs = group["SubjectID"][0] if "SubjectID" in group else None
            n_segments = len(ppg_refs)

            for seg_idx in range(n_segments):
                if MAX_SEGMENTS and len(y_sbp_list) >= MAX_SEGMENTS:
                    break

                ppg = load_segment(group, ppg_refs, seg_idx, "PPG_F")
                if len(ppg) == 0 or ppg.std() < MIN_PPG_STD:
                    flat_skipped += 1
                    continue

                # Extract features
                try:
                    feats = extract_ppg_features(ppg, sampling_rate=SAMPLING_RATE)
                except Exception:
                    skipped += 1
                    continue

                if len(feats) < 5:  # too few features → likely failed
                    skipped += 1
                    continue

                # Load labels
                try:
                    sbp = float(np.array(group[sbp_refs[seg_idx]]).flatten()[0])
                    dbp = float(np.array(group[dbp_refs[seg_idx]]).flatten()[0])
                except Exception:
                    skipped += 1
                    continue

                # Load demographics
                try:
                    age = float(np.array(group[age_refs[seg_idx]]).flatten()[0]) if age_refs is not None else np.nan
                    _gv = int(float(np.array(group[gender_refs[seg_idx]]).flatten()[0])) if gender_refs is not None else None
                    gender = chr(_gv) if _gv is not None else "unknown"  # ASCII: 70='F', 77='M'
                    subject_id = fname  # filename is the unique subject identifier
                except Exception:
                    age = np.nan
                    gender = "unknown"
                    subject_id = fname

                X_list.append(feats)
                y_sbp_list.append(sbp)
                y_dbp_list.append(dbp)
                segment_ids.append(f"{fname}#{seg_idx}")
                ages_list.append(age)
                genders_list.append(gender)
                subject_ids_list.append(subject_id)

            if MAX_SEGMENTS and len(y_sbp_list) >= MAX_SEGMENTS:
                break

        # Progress
        if (fi + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(valid_files)}] {len(y_sbp_list)} samples collected "
                  f"({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Collected: {len(y_sbp_list)} samples")
    print(f"  Skipped (feature fail): {skipped}")
    print(f"  Skipped (flat PPG): {flat_skipped}")

    if len(y_sbp_list) == 0:
        print("ERROR: No valid samples collected!")
        return

    # ── build feature matrix ────────────────────────────────────────────
    # Union all feature keys across all samples
    all_keys = sorted(set().union(*[set(f.keys()) for f in X_list]))
    print(f"\nFeature keys: {len(all_keys)}")
    for k in all_keys:
        print(f"  - {k}")

    X = np.zeros((len(X_list), len(all_keys)), dtype=np.float32)
    for i, feats in enumerate(X_list):
        for j, key in enumerate(all_keys):
            X[i, j] = feats.get(key, 0.0)

    y_sbp = np.array(y_sbp_list, dtype=np.float32)
    y_dbp = np.array(y_dbp_list, dtype=np.float32)

    # ── save ────────────────────────────────────────────────────────────
    out = {
        "X": X,
        "y_sbp": y_sbp,
        "y_dbp": y_dbp,
        "feature_names": all_keys,
        "segment_ids": segment_ids,
        "age": np.array(ages_list, dtype=np.float32),
        "gender": np.array(genders_list),
        "subjects": np.array(subject_ids_list),
        "stats": {
            "n_samples": len(y_sbp_list),
            "n_features": len(all_keys),
            "sbp_mean": float(np.mean(y_sbp)),
            "sbp_std": float(np.std(y_sbp)),
            "dbp_mean": float(np.mean(y_dbp)),
            "dbp_std": float(np.std(y_dbp)),
        },
    }

    out_path = os.path.join(OUTPUT_DIR, "pulsedb_features_full.npz")
    np.savez_compressed(out_path, **out)
    print(f"\nSaved to {out_path}")
    print(f"  X shape: {X.shape}")
    print(f"  SBP: mean={np.mean(y_sbp):.1f}, std={np.std(y_sbp):.1f}")
    print(f"  DBP: mean={np.mean(y_dbp):.1f}, std={np.std(y_dbp):.1f}")

    # Also save metadata as JSON
    meta_path = os.path.join(OUTPUT_DIR, "pulsedb_stats.json")
    with open(meta_path, "w") as f:
        json.dump(out["stats"], f, indent=2)
    print(f"  Stats saved to {meta_path}")


if __name__ == "__main__":
    main()
