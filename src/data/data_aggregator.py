import os
import h5py
import numpy as np
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
import pandas as pd
from tqdm import tqdm, trange

RAW_ROOT = root_dir / "data/raw/DroneRF"
OUT_FILE  = root_dir / "data/interim/dronerf.h5"
L = 100_000   # samples per segment, from MATLAB: L = 1e5

BUI_MAP = {
    "00000": "background",
    "10000": "bebop", "10001": "bebop",
    "10010": "bebop", "10011": "bebop",
    "10100": "ar",    "10101": "ar",
    "10110": "ar",    "10111": "ar",
    "11000": "phantom",
}

def read_raw_csv(csv_path: Path) -> np.ndarray:
    """Read 1-row × N-column CSV into a 1D float32 array."""
    raw = np.fromstring(open(csv_path).read(), sep=',', dtype=np.float32)
    return raw

def slice_segments(signal: np.ndarray, L: int) -> list[np.ndarray]:
    """
    following the original repo: https://github.com/Al-Sad/DroneRF/blob/master/Matlab/Main_1_Data_aggregation.m
    """
    n_segs = len(signal) // L
    return [signal[i*L:(i+1)*L] for i in range(n_segs)]

def build_h5(raw_root: Path, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    records = []
    seg_id = 0

    with h5py.File(out_file, "w") as hf:
        seg_grp = hf.create_group("segments")

        for bui, drone_type in tqdm(BUI_MAP.items()):
            # Find all CSV files matching this BUI
            csv_files = sorted(raw_root.rglob(f"{bui}*.csv"))
            if not csv_files:
                print(f"[WARN] No CSVs found for BUI {bui}")
                continue

            for csv_path in csv_files:
                # Parse band from filename: "11000H_0.csv" → "H"
                stem = csv_path.stem          # e.g. "11000H_0"
                band = "H" if "H" in stem[5:] else "L"
                file_idx = stem.split("_")[-1] # e.g. "0"

                signal = read_raw_csv(csv_path)
                segments = slice_segments(signal, L)

                for seg_within_file, seg_data in enumerate(segments):
                    key = f"{seg_id:05d}"
                    grp = seg_grp.create_group(key)
                    grp.create_dataset("signal", data=seg_data,
                                       compression="gzip", compression_opts=4)
                    grp.attrs.update({
                        "bui": bui, "drone_type": drone_type,
                        "band": band, "file_idx": file_idx,
                        "seg_within_file": seg_within_file,
                        "n_samples": len(seg_data),
                    })
                    records.append({
                        "segment_id": key, "bui": bui,
                        "drone_type": drone_type, "band": band,
                        "file_idx": file_idx,
                        "seg_within_file": seg_within_file,
                    })
                    seg_id += 1

            print(f"  BUI {bui} ({drone_type}): done")

        # Metadata as queryable dataset
        df = pd.DataFrame(records)
        for col in df.columns:
            hf.create_dataset(f"metadata/{col}",
                              data=df[col].values.astype("S" if df[col].dtype == object else df[col].dtype))

    df.to_parquet(out_file.with_suffix(".meta.parquet"), index=False)
    print(f"\n✓ {seg_id} segments written → {out_file}")

def inspect_h5(h5_path: Path):
    """Print H5 hierarchy, segment count, and a sample segment."""
    with h5py.File(h5_path, "r") as hf:


        # 2. Segment count + label distribution
        print("\n── Segment Summary ───────────────────────────")
        n_segs = len(hf["segments"])
        print(f"  Total segments : {n_segs}")

        drone_types = hf["metadata/drone_type"][:].astype(str)
        unique, counts = np.unique(drone_types, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  {label:12s} : {count} segments")

        # 3. One sample segment
        print("\n── Sample Segment (00000) ────────────────────")
        seg = hf["segments/00000"]
        print(f"  signal shape   : {seg['signal'].shape}")
        print(f"  signal dtype   : {seg['signal'].dtype}")
        print(f"  signal range   : [{seg['signal'][:].min():.4f}, {seg['signal'][:].max():.4f}]")
        print(f"  attrs          : {dict(seg.attrs)}")


if __name__ == "__main__":
    if not OUT_FILE.exists():
        build_h5(RAW_ROOT, OUT_FILE)
    inspect_h5(OUT_FILE)