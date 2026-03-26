# -*- coding: utf-8 -*-
"""
data_aggregator.py
==================
Builds a self-contained HDF5 archive from raw DroneRF CSV files.

Changes from v1:
  - signal_format="interleaved_iq" stored at file level and per-segment.
    Downstream dsp_transforms.py reads this attr instead of guessing.
  - recording_id attr links L-band and H-band segments from the same
    physical recording (bui + file_idx + seg_within_file). Required for
    any fusion experiment and for correct group-aware train/test splitting.
  - fs_hz stored per segment (H=40e6, L=10e6) — they differ.
  - label (int) stored per segment alongside drone_type string.
  - inspect_h5() now prints the full H5 hierarchy (was empty before).
  - build_splits() generates and saves a fixed train/val/test split to
    splits.json, grouped by recording_id so no physical recording leaks
    across splits. Run once; all experiments reuse these indices.
"""

import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

root_dir = Path(__file__).resolve().parent.parent.parent

RAW_ROOT   = root_dir / "data/raw/DroneRF"
OUT_FILE   = root_dir / "data/interim/dronerf.h5"
SPLIT_FILE = root_dir / "data/interim/dronerf_splits.json"

L: int = 100_000   # samples per segment, from MATLAB: L = 1e5

# H-band: 40 MHz, L-band: 10 MHz  (USRP B210, from the paper)
BAND_FS: dict[str, float] = {"H": 40e6, "L": 10e6}

BUI_MAP: dict[str, str] = {
    "00000": "background",
    "10000": "bebop",  "10001": "bebop",
    "10010": "bebop",  "10011": "bebop",
    "10100": "ar",     "10101": "ar",
    "10110": "ar",     "10111": "ar",
    "11000": "phantom",
}

# Integer label for classification — stable across runs
LABEL_MAP: dict[str, int] = {
    "background": 0,
    "bebop":      1,
    "ar":         2,
    "phantom":    3,
}

VAL_FRAC:    float = 0.10   # fraction of train set held out for validation
TEST_FRAC:   float = 0.15   # fraction of full dataset held out as test
RANDOM_SEED: int   = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_raw_csv(csv_path: Path) -> np.ndarray:
    """Read 1-row × N-column CSV into a 1D float32 array."""
    raw = np.fromstring(open(csv_path).read(), sep=",", dtype=np.float32)
    return raw


def slice_segments(signal: np.ndarray, seg_len: int) -> list[np.ndarray]:
    """
    Non-overlapping fixed-length segments, following:
    https://github.com/Al-Sad/DroneRF/blob/master/Matlab/Main_1_Data_aggregation.m
    Tail samples that don't fill a full segment are discarded.
    """
    n_segs = len(signal) // seg_len
    return [signal[i * seg_len : (i + 1) * seg_len] for i in range(n_segs)]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_h5(raw_root: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    records = []
    seg_id = 0

    with h5py.File(out_file, "w") as hf:

        # File-level metadata — lets any reader know the storage convention
        # without opening a segment first.
        hf.attrs["signal_format"]   = "interleaved_iq"   # IQIQIQ... float32
        hf.attrs["segment_length"]  = L
        hf.attrs["dataset"]         = "DroneRF"
        hf.attrs["label_map"]       = json.dumps(LABEL_MAP)

        seg_grp = hf.create_group("segments")

        for bui, drone_type in tqdm(BUI_MAP.items(), desc="BUI"):
            csv_files = sorted(raw_root.rglob(f"{bui}*.csv"))
            if not csv_files:
                print(f"[WARN] No CSVs found for BUI {bui}")
                continue

            for csv_path in csv_files:
                stem     = csv_path.stem                   # e.g. "11000H_0"
                band     = "H" if "H" in stem[5:] else "L"
                file_idx = stem.split("_")[-1]             # e.g. "0"
                fs_hz    = BAND_FS[band]

                signal   = read_raw_csv(csv_path)
                segments = slice_segments(signal, L)

                for seg_within_file, seg_data in enumerate(segments):
                    key = f"{seg_id:05d}"
                    grp = seg_grp.create_group(key)

                    grp.create_dataset(
                        "signal", data=seg_data,
                        compression="gzip", compression_opts=4,
                    )

                    # recording_id: uniquely identifies one physical recording
                    # session. L-band and H-band segments captured at the same
                    # time share the same recording_id. Use this for:
                    #   (a) band-fusion experiments (join on recording_id)
                    #   (b) GroupShuffleSplit so no recording leaks across splits
                    recording_id = f"{bui}_{file_idx}_{seg_within_file}"

                    grp.attrs.update({
                        # Provenance
                        "bui":             bui,
                        "drone_type":      drone_type,
                        "label":           LABEL_MAP[drone_type],  # int
                        "band":            band,
                        "fs_hz":           fs_hz,
                        "file_idx":        file_idx,
                        "seg_within_file": seg_within_file,
                        "n_samples":       len(seg_data),
                        # IQ convention — tells dsp_transforms.to_complex() which mode
                        "signal_format":   "interleaved_iq",
                        # Cross-band linkage
                        "recording_id":    recording_id,
                    })

                    records.append({
                        "segment_id":      key,
                        "bui":             bui,
                        "drone_type":      drone_type,
                        "label":           LABEL_MAP[drone_type],
                        "band":            band,
                        "fs_hz":           fs_hz,
                        "file_idx":        file_idx,
                        "seg_within_file": seg_within_file,
                        "recording_id":    recording_id,
                    })
                    seg_id += 1

            print(f"  BUI {bui} ({drone_type}): done")

        # Metadata queryable inside HDF5 (kept for parity with v1)
        df = pd.DataFrame(records)
        for col in df.columns:
            hf.create_dataset(
                f"metadata/{col}",
                data=df[col].values.astype("S" if df[col].dtype == object else df[col].dtype),
            )

    # Flat parquet — faster to query than HDF5 metadata for split building
    df.to_parquet(out_file.with_suffix(".meta.parquet"), index=False)
    print(f"\n✓ {seg_id} segments written → {out_file}")


# ---------------------------------------------------------------------------
# Split generation
# ---------------------------------------------------------------------------

def build_splits(meta_parquet: Path, split_file: Path) -> None:
    """
    Generate train / val / test splits and save to JSON.

    Splitting strategy:
      - Stratified by label so class distribution is preserved.
      - Grouped by recording_id so no physical recording session appears
        in more than one split. This prevents data leakage that would occur
        if consecutive segments from the same capture ended up in both
        train and test.
      - Run once; all downstream experiments load from split_file.

    Output JSON structure:
      {
        "train": ["00012", "00045", ...],   # segment_id strings
        "val":   ["00003", ...],
        "test":  ["00078", ...],
        "meta":  { ... provenance ... }
      }
    """
    df = pd.read_parquet(meta_parquet)

    # Collapse to one row per recording_id for group-level splitting
    groups = (
        df.groupby("recording_id")
        .agg(label=("label", "first"), segment_ids=("segment_id", list))
        .reset_index()
    )

    # First cut: hold out test set
    train_val_idx, test_idx = train_test_split(
        groups.index,
        test_size=TEST_FRAC,
        stratify=groups["label"],
        random_state=RANDOM_SEED,
    )

    # Second cut: carve val from the remaining train pool
    val_size_adjusted = VAL_FRAC / (1.0 - TEST_FRAC)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size_adjusted,
        stratify=groups.loc[train_val_idx, "label"],
        random_state=RANDOM_SEED,
    )

    def collect_ids(idx):
        return sorted(sid for i in idx for sid in groups.loc[i, "segment_ids"])

    train_ids = collect_ids(train_idx)
    val_ids   = collect_ids(val_idx)
    test_ids  = collect_ids(test_idx)

    splits = {
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids,
        "meta": {
            "random_seed":   RANDOM_SEED,
            "test_frac":     TEST_FRAC,
            "val_frac":      VAL_FRAC,
            "n_train":       len(train_ids),
            "n_val":         len(val_ids),
            "n_test":        len(test_ids),
            "split_by":      "recording_id",
            "stratified_by": "label",
        },
    }

    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(splits, indent=2))
    print(f"\n✓ Splits saved → {split_file}")
    print(f"  train : {splits['meta']['n_train']} segments")
    print(f"  val   : {splits['meta']['n_val']}   segments")
    print(f"  test  : {splits['meta']['n_test']}  segments")


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect_h5(h5_path: Path) -> None:
    """Print full H5 hierarchy, file-level attrs, label distribution, and one sample segment."""
    with h5py.File(h5_path, "r") as hf:

        # 1. File-level attrs
        print("\n── File Attributes ───────────────────────────")
        for k, v in hf.attrs.items():
            print(f"  {k:<20} : {v}")

        # 2. H5 hierarchy (top 2 levels — avoids printing every segment key)
        print("\n── H5 Hierarchy (top 2 levels) ───────────────")
        def _visitor(name, obj):
            depth = name.count("/")
            if depth < 2:
                kind = "GROUP" if isinstance(obj, h5py.Group) else "DATASET"
                print(f"  {'  ' * depth}{name}  [{kind}]")
        hf.visititems(_visitor)

        # 3. Segment count + label distribution
        print("\n── Segment Summary ───────────────────────────")
        n_segs = len(hf["segments"])
        print(f"  Total segments : {n_segs}")
        drone_types = hf["metadata/drone_type"][:].astype(str)
        unique, counts = np.unique(drone_types, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label:12s} : {count:5d} segments  ({100 * count / n_segs:.1f}%)")

        # 4. Band distribution
        print("\n── Band Distribution ─────────────────────────")
        bands = hf["metadata/band"][:].astype(str)
        for b, c in zip(*np.unique(bands, return_counts=True)):
            print(f"  Band {b}        : {c:5d} segments")

        # 5. One sample segment — full audit
        print("\n── Sample Segment (00000) ────────────────────")
        seg = hf["segments/00000"]
        sig = seg["signal"][:]
        print(f"  signal shape   : {seg['signal'].shape}")
        print(f"  signal dtype   : {seg['signal'].dtype}")
        print(f"  signal range   : [{sig.min():.4f}, {sig.max():.4f}]")
        print(f"  signal mean    : {sig.mean():.6f}  (should be ~0 for RF baseband)")
        print(f"  signal std     : {sig.std():.4f}")
        print(f"  first 8 values : {sig[:8]}")
        print(f"  attrs:")
        for k, v in seg.attrs.items():
            print(f"    {k:<20} : {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not OUT_FILE.exists():
        build_h5(RAW_ROOT, OUT_FILE)
    else:
        print(f"[INFO] H5 already exists: {OUT_FILE}  (delete to rebuild)")

    inspect_h5(OUT_FILE)

    if not SPLIT_FILE.exists():
        build_splits(OUT_FILE.with_suffix(".meta.parquet"), SPLIT_FILE)
    else:
        print(f"\n[INFO] Splits already exist: {SPLIT_FILE}  (delete to regenerate)")

    df = pd.read_parquet(OUT_FILE.with_suffix(".meta.parquet"))
    print(f"\n── Parquet Head ──────────────────────────────")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")