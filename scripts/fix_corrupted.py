#!/usr/bin/env python3
"""
Corrupted PulseDB file re-downloader.
Run this script on a machine with Google Drive access (e.g., Windows + VPN).

It identifies which .mat files are corrupted and generates:
  1. A list of missing files
  2. Download commands (gdown for Google Drive)

Instructions:
  1. pip install gdown tqdm
  2. python3 fix_corrupted.py --list     # list corrupted files
  3. python3 fix_corrupted.py --download  # download missing files
"""

import os, sys, glob, argparse, subprocess
import h5py

DATA_DIR = os.path.expanduser("~/projects/bishe/data/pulsedb/Segment_Files")

# Google Drive folder ID from PulseDB README
GDRIVE_FOLDER = "10mz4mfBo6NczPNbbjX0a9tAKQSMugBjV"


def find_corrupted(data_dir: str):
    """Return list of corrupted .mat files."""
    corrupted = []
    total = 0
    for root, dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.endswith(".mat"):
                continue
            total += 1
            fpath = os.path.join(root, fname)
            try:
                with h5py.File(fpath, "r") as f:
                    _ = f["Subj_Wins"]["PPG_F"]
            except Exception:
                corrupted.append(fpath)
    return corrupted, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List corrupted files")
    parser.add_argument("--download", action="store_true", help="Download missing files")
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    corrupted, total = find_corrupted(args.data_dir)
    good = total - len(corrupted)

    print(f"Total files: {total}")
    print(f"Good: {good}")
    print(f"Corrupted: {len(corrupted)}")

    if args.list or not args.download:
        if corrupted:
            print("\nCorrupted files:")
            for fp in corrupted:
                subdir = "MIMIC" if "MIMIC" in fp else "VitalDB"
                print(f"  [{subdir}] {os.path.basename(fp)}")

    if args.download:
        print(f"\nDownloading {len(corrupted)} files from Google Drive...")
        print("This requires gdown: pip install gdown")
        print()

        for fp in corrupted:
            fname = os.path.basename(fp)
            subdir = "PulseDB_MIMIC" if "MIMIC" in fp else "PulseDB_Vital"
            dest_dir = os.path.join(args.data_dir, subdir)
            os.makedirs(dest_dir, exist_ok=True)

            # Google Drive direct download via gdown
            # gdown can search by filename in a shared folder
            cmd = [
                "gdown", "--fuzzy",
                f"--folder", GDRIVE_FOLDER,
                f"--output", os.path.join(dest_dir, fname),
                f"--remaining-ok",
            ]
            print(f"  Downloading {fname}...")
            try:
                subprocess.run(cmd, check=True, timeout=300)
            except Exception as e:
                print(f"    Failed: {e}")

    if not args.list and not args.download:
        parser.print_help()


if __name__ == "__main__":
    main()
