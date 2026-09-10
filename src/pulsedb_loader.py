"""
PulseDB data loader for v2.0 dataset (MATLAB v7.3 / HDF5 format).
Supports loading individual segment files and iterating over batches.

Dataset structure per .mat file:
  Subj_Wins/
    ├── PPG_F, ECG_F, ABP_F        # Filtered signals (shape: 1×1250)
    ├── PPG_Raw, ECG_Raw, ABP_Raw   # Raw signals (shape: 1×1250)
    ├── PPG_Record, ECG_Record      # Non-normalized raw (shape: 1×1250)
    ├── PPG_Record_F, ECG_Record_F  # Non-normalized filtered (shape: 1×1250)
    ├── SegSBP, SegDBP              # Blood pressure labels (shape: 1×1)
    ├── Age, Gender, Height, Weight, BMI  # Demographics (shape: 1×1)
    └── SubjectID, SegmentID, WinID, T    # Metadata

Reference: github.com/pulselabteam/PulseDB
"""

import os
import numpy as np
import h5py
from typing import Dict, Optional, Iterator, Tuple


DATA_DIR = "/home/lby/projects/bishe/data/pulsedb"

# Mapping from standard keys to PulseDB field names in Subj_Wins group
_SIGNAL_KEY_MAP = {
    "ecg": "ECG_F",
    "ppg": "PPG_F",
    "abp": "ABP_F",
}
_EXTRA_KEY_MAP = {
    "ecg_raw": "ECG_Raw",
    "ppg_raw": "PPG_Raw",
    "abp_raw": "ABP_Raw",
    "sbp": "SegSBP",
    "dbp": "SegDBP",
    "age": "Age",
    "gender": "Gender",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "subject_id": "SubjectID",
    "segment_id": "SegmentID",
}


def _count_segments_v73(filepath: str) -> int:
    """Count the number of segments in a MATLAB v7.3 file.

    Single-segment files return 1. Multi-segment files return the
    number of HDF5 references found in the first signal field.
    """
    with h5py.File(filepath, "r") as f:
        group = f.get("Subj_Wins", f)
        for k in group.keys():
            if hasattr(group[k], 'dtype') and not k.startswith('#'):
                ds = group[k]
                if ds.dtype.kind == 'O':
                    try:
                        refs = ds[0]
                        return len(refs)
                    except Exception:
                        break
                break
    return 1


def _read_mat_v73(filepath: str, segment_index: int = 0) -> Dict[str, np.ndarray]:
    """Read MATLAB v7.3 (HDF5-based) .mat file.

    Some files contain multiple segments stored as HDF5 references.
    segment_index selects which segment to load (default: first).
    """
    data = {}
    with h5py.File(filepath, "r") as f:
        group = f.get("Subj_Wins", f)

        # Detect if fields contain HDF5 references (multi-segment file)
        first_field = None
        for k in group.keys():
            if hasattr(group[k], 'dtype') and not k.startswith('#'):
                first_field = k
                break

        if first_field is None:
            return data

        ds = group[first_field]
        is_ref = ds.dtype.kind == 'O'

        for std_key, field_name in {**_SIGNAL_KEY_MAP, **_EXTRA_KEY_MAP}.items():
            if field_name not in group:
                continue

            ds = group[field_name]

            if is_ref:
                # Multi-segment file: dereference the HDF5 reference at segment_index
                try:
                    refs = ds[0]  # shape (1, N) -> take first row
                    if segment_index >= len(refs):
                        continue
                    resolved = group[refs[segment_index]]
                    arr = np.array(resolved).flatten().astype(np.float64)
                except (ValueError, TypeError, IndexError):
                    continue
            else:
                # Single-segment file: read directly
                if ds.dtype.kind in ('O', 'V'):
                    try:
                        arr = np.array(ds).flatten()
                        if arr.dtype.kind == 'O':
                            continue
                        data[std_key] = arr.astype(np.float64)
                    except (ValueError, TypeError):
                        continue
                    continue
                arr = np.array(ds).flatten().astype(np.float64)

            data[std_key] = arr

    return data


def _read_mat_legacy(filepath: str) -> Dict[str, np.ndarray]:
    """Read old-style MATLAB .mat file via scipy (non-HDF5)."""
    import scipy.io as sio
    raw = sio.loadmat(filepath)
    data = {}
    for std_key, field_name in _SIGNAL_KEY_MAP.items():
        if field_name in raw:
            data[std_key] = np.array(raw[field_name]).flatten().astype(np.float64)
    return data


def load_pulsedb_sample(filepath: str, segment_index: int = 0) -> Dict[str, np.ndarray]:
    """
    Load a PulseDB segment from a .mat file.

    Args:
        filepath: Path to .mat (v7.3 or legacy), .h5, or .npy file.
        segment_index: For multi-segment files, which segment to load (0-based).

    Returns:
        Dict with at minimum: "ecg", "ppg", "abp" — each a 1D float64 array.
        Also includes: "sbp", "dbp", "age", "gender", etc. if available.
    """
    if filepath.endswith(".h5") or filepath.endswith(".hdf5"):
        with h5py.File(filepath, "r") as f:
            data = {}
            for std_key, field_name in _SIGNAL_KEY_MAP.items():
                if field_name in f:
                    data[std_key] = np.array(f[field_name]).flatten().astype(np.float64)
            return data

    if filepath.endswith(".mat"):
        # Detect MATLAB version from magic bytes
        with open(filepath, "rb") as fh:
            magic = fh.read(20)
        # v7.3 files start with "MATLAB 7.3 MAT-file" and are HDF5-based
        if b"MATLAB 7.3" in magic:
            return _read_mat_v73(filepath, segment_index)
        # Legacy v5/v7 files start with "MATLAB 5.0 MAT-file"
        if magic[:6] == b"MATLAB":
            return _read_mat_legacy(filepath)
        # Fallback: try v7.3 first (more common), then legacy
        try:
            return _read_mat_v73(filepath, segment_index)
        except Exception:
            return _read_mat_legacy(filepath)

    if filepath.endswith(".npy"):
        data = np.load(filepath, allow_pickle=True).item()
        if isinstance(data, np.ndarray):
            return {"ecg": data}
        return data

    return {}


def count_pulsedb_segments(filepath: str) -> int:
    """
    Return the number of segments stored in a PulseDB .mat file.

    Single-segment files return 1. Multi-segment files return the
    number of HDF5 references embedded in the Subj_Wins group.
    """
    if filepath.endswith(".mat"):
        with open(filepath, "rb") as fh:
            magic = fh.read(20)
        if b"MATLAB 7.3" in magic:
            return _count_segments_v73(filepath)
    return 1


def iter_pulsedb_segments(filepath: str) -> Iterator[Dict[str, np.ndarray]]:
    """
    Yield all segments from a PulseDB .mat file.

    For single-segment files, yields one dict.
    For multi-segment files, dereferences each HDF5 reference and yields
    each segment's signals + labels.
    """
    n_segments = count_pulsedb_segments(filepath)
    if n_segments <= 1:
        data = load_pulsedb_sample(filepath, segment_index=0)
        if data:
            yield data
        return

    for idx in range(n_segments):
        data = load_pulsedb_sample(filepath, segment_index=idx)
        if data and "ppg" in data and "ecg" in data:
            yield data


def load_pulsedb_batch(
    segment_ids: Optional[list] = None,
    max_samples: int = 100,
) -> Iterator[Tuple[str, Dict[str, np.ndarray]]]:
    """
    Iterate over PulseDB segments.

    Args:
        segment_ids: Specific segment IDs to load. If None, loads first max_samples.
        max_samples: Maximum number of segments to load.

    Yields:
        (segment_id, signals_dict) tuples
    """
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"PulseDB data directory not found: {DATA_DIR}\n"
            "Download from: https://physionet.org/content/pulsedb/"
        )

    count = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for fname in sorted(files):
            if fname.endswith((".h5", ".mat", ".npy")):
                if segment_ids and fname not in segment_ids:
                    continue
                filepath = os.path.join(root, fname)
                try:
                    signals = load_pulsedb_sample(filepath)
                    yield fname, signals
                    count += 1
                    if count >= max_samples:
                        return
                except Exception as e:
                    print(f"Warning: failed to load {fname}: {e}")
                    continue


def load_pulsedb_metadata() -> Optional[Dict]:
    """Load PulseDB subject metadata if available."""
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    if os.path.exists(metadata_path):
        import csv
        metadata = {}
        with open(metadata_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata[row.get("subject_id", row.get("segment_id", ""))] = row
        return metadata
    return None


if __name__ == "__main__":
    print(f"PulseDB data directory: {DATA_DIR}")
    print(f"Exists: {os.path.isdir(DATA_DIR)}")
    if os.path.isdir(DATA_DIR):
        contents = os.listdir(DATA_DIR)
        print(f"Contents ({len(contents)} items): {contents[:20]}")
    else:
        print("To download PulseDB:")
        print("  1. Visit: https://physionet.org/content/pulsedb/")
        print("  2. Create a PhysioNet account")
        print("  3. Request access to the dataset")
        print("  4. Download to:", DATA_DIR)
