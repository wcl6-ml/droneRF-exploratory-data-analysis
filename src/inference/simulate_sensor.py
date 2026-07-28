"""
simulate_sensor.py
===================
Replays a real DroneRF .h5 archive as if it were a live sensor feed.



Output contract (what the next script depends on)
---------------------------------------------------
This module exposes one generator: `replay(...)`. Each item it yields is
a dict:

    {
        "recording_id":    str,
        "seg_within_file": int,
        "drone_type":      str,     # ground truth, human-readable
        "label":           int,     # ground truth, int form
        "raw_h":           np.ndarray (float32, interleaved I/Q),
        "raw_l":           np.ndarray (float32, interleaved I/Q),
        "fs_h":            float,
        "fs_l":            float,
        "emitted_at":      float,   # time.monotonic() when yielded
    }

"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Small helper -- h5py attr string decoding is inconsistent across versions
# ---------------------------------------------------------------------------

def _decode(v):
    return v.decode("utf-8") if isinstance(v, bytes) else v


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _load_split_ids(split_file: Path, split: str) -> set[str]:
    splits = json.loads(split_file.read_text())
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Available: {list(splits.keys())}")
    return set(splits[split])


def _build_recording_index(h5_path: Path, allowed_segment_ids: set[str]) -> dict:
    """
    Groups segments by recording_id, restricted to the given split's
    segment ids. recording_id links H and L band segments captured from
    the same physical recording (see data_aggregator.py).

    Returns:
        {
          recording_id: {
            "drone_type": str,
            "label": int,
            "bands": {"H": {seg_within_file: segment_key}, "L": {...}},
            "paired_positions": [seg_within_file, ...],  # positions with both bands present
          },
          ...
        }
    """
    index: dict = defaultdict(lambda: {"drone_type": None, "label": None, "bands": {"H": {}, "L": {}}})

    with h5py.File(h5_path, "r") as hf:
        for seg_key in hf["segments"]:
            if seg_key not in allowed_segment_ids:
                continue
            attrs = hf["segments"][seg_key].attrs
            rec_id = _decode(attrs["recording_id"])
            band = _decode(attrs["band"])
            entry = index[rec_id]
            entry["drone_type"] = _decode(attrs["drone_type"])
            entry["label"] = int(attrs["label"])
            entry["bands"][band][int(attrs["seg_within_file"])] = seg_key

    # Keep only recordings where every position has BOTH bands --
    # predictor.predict() needs raw_h and raw_l together, always.
    complete = {}
    for rec_id, entry in index.items():
        paired = sorted(set(entry["bands"]["H"]) & set(entry["bands"]["L"]))
        if not paired:
            continue
        entry["paired_positions"] = paired
        complete[rec_id] = entry

    return complete


def list_available_classes(recording_index: dict) -> list[str]:
    return sorted({entry["drone_type"] for entry in recording_index.values()})


# ---------------------------------------------------------------------------
# Window loading + real-time pacing
# ---------------------------------------------------------------------------

def _load_window(hf: h5py.File, rec_id: str, entry: dict, position: int) -> dict:
    h_seg = hf["segments"][entry["bands"]["H"][position]]
    l_seg = hf["segments"][entry["bands"]["L"][position]]

    return {
        "recording_id": rec_id,
        "seg_within_file": position,
        "drone_type": entry["drone_type"],
        "label": entry["label"],
        "raw_h": h_seg["signal"][:],
        "raw_l": l_seg["signal"][:],
        "fs_h": float(h_seg.attrs["fs_hz"]),
        "fs_l": float(l_seg.attrs["fs_hz"]),
    }


def _real_time_duration(raw_h: np.ndarray, fs_h: float) -> float:
    """Physical duration of one window in seconds, derived from the H-band capture."""
    n_complex_samples = len(raw_h) // 2  # interleaved I/Q -> half as many complex samples
    return n_complex_samples / fs_h


# ---------------------------------------------------------------------------
# Replay ordering
# ---------------------------------------------------------------------------

def _scenario_order(recording_index: dict, scenario: list[str], seed: Optional[int]) -> list[str]:
    """Picks one recording_id per scenario entry, in the order requested."""
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for rec_id, entry in recording_index.items():
        by_class[entry["drone_type"]].append(rec_id)
    for ids in by_class.values():
        rng.shuffle(ids)

    used: dict[str, int] = defaultdict(int)  # cursor per class; cycles if scenario repeats a class more than we have recordings
    ordered_rec_ids = []
    for drone_type in scenario:
        pool = by_class.get(drone_type)
        if not pool:
            raise ValueError(
                f"No recordings of class '{drone_type}' in this split. "
                f"Available classes: {sorted(by_class)}"
            )
        idx = used[drone_type] % len(pool)
        ordered_rec_ids.append(pool[idx])
        used[drone_type] += 1
    return ordered_rec_ids


def _shuffled_order(recording_index: dict, seed: Optional[int]) -> list[str]:
    ids = sorted(recording_index)
    if seed is not None:
        random.Random(seed).shuffle(ids)
    return ids


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def replay(
    h5_path: Path,
    split_file: Path,
    split: str = "test",
    mode: str = "scenario",
    scenario: Optional[list[str]] = None,
    speed_factor: float = 1.0,
    seed: Optional[int] = 42,
) -> Iterator[dict]:
    """
    Yields one window dict at a time, paced to simulate a live sensor.

    mode="scenario": replays `scenario` (ordered list of drone_type
        strings). Each entry is backed by one full real recording,
        played start to finish in original order.
    mode="shuffled": replays every recording in the split, raw order.
        Expect chaotic class-switching between windows -- that's a
        faithful replay of how the split was built, not a defect here.

    speed_factor: 1.0 = real time (can be very slow -- RF windows are
        physically microseconds to milliseconds long). 20.0 = 20x
        faster than real capture rate; use this for demos.
    """
    allowed_ids = _load_split_ids(split_file, split)
    recording_index = _build_recording_index(h5_path, allowed_ids)

    if mode == "scenario":
        if not scenario:
            raise ValueError("mode='scenario' requires a non-empty `scenario` list of drone_type strings")
        rec_order = _scenario_order(recording_index, scenario, seed)
    elif mode == "shuffled":
        rec_order = _shuffled_order(recording_index, seed)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'scenario' or 'shuffled'.")

    with h5py.File(h5_path, "r") as hf:
        for rec_id in rec_order:
            entry = recording_index[rec_id]
            for position in entry["paired_positions"]:
                window = _load_window(hf, rec_id, entry, position)
                duration = _real_time_duration(window["raw_h"], window["fs_h"])
                window["emitted_at"] = time.monotonic()
                yield window
                time.sleep(max(0.0, duration / speed_factor))


# ---------------------------------------------------------------------------
# Standalone run -- validates this script alone, before anything downstream exists
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Replay dronerf.h5 as a simulated live sensor feed.")
    parser.add_argument("--h5", default="data/interim/dronerf.h5")
    parser.add_argument("--splits", default="data/interim/dronerf_splits.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", default="scenario", choices=["scenario", "shuffled"])
    parser.add_argument(
        "--scenario",
        default="background,bebop,background,phantom,background,ar",
        help="Comma-separated drone_type sequence, e.g. background,bebop,background",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = real time)")
    parser.add_argument("--list-classes", action="store_true", help="Print available classes for this split and exit")
    args = parser.parse_args()

    h5_path = Path(args.h5)
    split_file = Path(args.splits)

    if args.list_classes:
        allowed_ids = _load_split_ids(split_file, args.split)
        idx = _build_recording_index(h5_path, allowed_ids)
        print(f"Classes available in split '{args.split}': {list_available_classes(idx)}")
        # return

    scenario = args.scenario.split(",") if args.mode == "scenario" else None

    stream = replay(
        h5_path=h5_path,
        split_file=split_file,
        split=args.split,
        mode=args.mode,
        scenario=scenario,
        speed_factor=args.speed,
    )

    print(f"Replaying split='{args.split}' mode='{args.mode}' speed={args.speed}x ...\n")
    for i, window in enumerate(stream):
        print(
            f"[{i:04d}] rec={window['recording_id']:<20} "
            f"pos={window['seg_within_file']:<3} "
            f"truth={window['drone_type']:<12} "
            f"raw_h={window['raw_h'].shape} raw_l={window['raw_l'].shape}"
        )


if __name__ == "__main__":
    main()