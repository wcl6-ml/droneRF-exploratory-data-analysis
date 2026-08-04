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
import requests

from tqdm import tqdm
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


def class_recording_stats(recording_index: dict) -> dict[str, dict]:
    """
    Per-class visibility into the replay pool: how many distinct real
    recordings back each class, and how many clips (paired positions)
    each recording contributes on average. This matters directly for
    long scenario runs -- "background:20000" against a pool of 40
    recordings means ~500x reuse per recording, which you'll want to
    know before staring at a Prometheus dashboard wondering why the
    pattern looks periodic.
    """
    stats: dict[str, dict] = defaultdict(lambda: {"recordings": 0, "total_clips": 0})
    for entry in recording_index.values():
        s = stats[entry["drone_type"]]
        s["recordings"] += 1
        s["total_clips"] += len(entry["paired_positions"])
    return dict(sorted(stats.items()))


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

def _scenario_order(
    recording_index: dict,
    scenario: list[str],
    seed: Optional[int],
    reshuffle_on_exhaust: bool = True,
) -> list[str]:
    """
    Picks one recording_id per scenario entry, in the order requested.

    When a class's recording pool runs out (scenario asks for more
    instances of a class than there are distinct real recordings), we
    wrap around and reuse recordings. reshuffle_on_exhaust controls what
    "wrap around" means:

      True  (default): reshuffle the pool each time it's exhausted, so
          repeats still happen (finite real data -- unavoidable) but the
          *order* of repeats isn't identical lap after lap. Avoids a
          mechanically periodic pattern showing up on a long-running
          dashboard, which would look like a bug even though it isn't.
      False: cycle through the same fixed order every lap. Useful if you
          want a literally reproducible, inspectable sequence.
    """
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = defaultdict(list)
    for rec_id, entry in recording_index.items():
        by_class[entry["drone_type"]].append(rec_id)
    for ids in by_class.values():
        rng.shuffle(ids)

    cursors: dict[str, int] = defaultdict(int)  # next index to hand out, per class
    ordered_rec_ids = []
    for drone_type in tqdm(scenario):
        pool = by_class.get(drone_type)
        if not pool:
            raise ValueError(
                f"No recordings of class '{drone_type}' in this split. "
                f"Available classes: {sorted(by_class)}"
            )
        idx = cursors[drone_type]
        if idx >= len(pool):
            if reshuffle_on_exhaust:
                rng.shuffle(pool)
            idx = 0
        ordered_rec_ids.append(pool[idx])
        cursors[drone_type] = idx + 1
    return ordered_rec_ids


def parse_scenario_spec(spec: str) -> list[str]:
    """
    Parses a compact scenario spec into an expanded, ordered list of
    drone_type labels -- the form `replay()` actually consumes.

    Two token forms, comma-separated, freely mixable:
      "background"        -> one occurrence
      "background:20000"  -> 20000 consecutive occurrences

    e.g. "background:20000,ar:10000,background:5000" expands to a list
    of 35000 labels: 20000 "background", then 10000 "ar", then 5000 more
    "background" -- three ordered blocks, not interleaved. This is what
    makes long monitored runs practical without spelling out every
    repeat by hand.
    """
    expanded: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            cls, count_str = token.split(":", 1)
            cls = cls.strip()
            try:
                count = int(count_str.strip())
            except ValueError:
                raise ValueError(f"Invalid count in scenario token '{token}' -- expected 'class:integer'")
            if count <= 0:
                raise ValueError(f"Count must be positive in scenario token '{token}'")
            expanded.extend([cls] * count)
        else:
            expanded.append(token)
    return expanded


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
    reshuffle_on_exhaust: bool = True,
) -> Iterator[dict]:
    """
    Yields one window dict at a time, paced to simulate a live sensor.

    mode="scenario": replays `scenario` (ordered list of drone_type
        strings -- use parse_scenario_spec() to build this from a
        compact "class:count" string). Each entry is backed by one full
        real recording, played start to finish in original order.
    mode="shuffled": replays every recording in the split, raw order.
        Expect chaotic class-switching between windows -- that's a
        faithful replay of how the split was built, not a defect here.

    speed_factor: 1.0 = real time (can be very slow -- RF windows are
        physically microseconds to milliseconds long). 20.0 = 20x
        faster than real capture rate; use this for demos.
    reshuffle_on_exhaust: see _scenario_order(). Only applies in
        mode="scenario"; ignored in mode="shuffled".
    """
    allowed_ids = _load_split_ids(split_file, split)
    recording_index = _build_recording_index(h5_path, allowed_ids)

    if mode == "scenario":
        if not scenario:
            raise ValueError("mode='scenario' requires a non-empty `scenario` list of drone_type strings")
        rec_order = _scenario_order(recording_index, scenario, seed, reshuffle_on_exhaust)
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
# HTTP client -- the real integration test. Exercises the same
# JSON-over-HTTP path a real edge client would use to talk to a running
# `uvicorn src.api.main:app` instance, instead of calling service.py
# in-process. This is what finally proves predictor.py + service.py +
# main.py + labels.json agree with each other end to end.
# ---------------------------------------------------------------------------
 
def check_api_health(api_url: str, timeout: float = 5.0) -> None:
    """
    Fails fast, with a clear message, if the API isn't up or the model
    didn't load -- rather than letting the first /v1/predict call raise
    a generic connection error mid-scenario.
    """
    resp = requests.get(f"{api_url}/health", timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("model_loaded"):
        raise RuntimeError(f"API at {api_url} is up but model_loaded=False: {body}")
 
 
def send_window(window: dict, api_url: str, timeout: float = 5.0) -> dict:
    """
    POSTs one window's raw_h/raw_l to a running /v1/predict endpoint and
    returns the response with ground truth attached, so the caller can
    compare predicted vs. actual without threading extra state around.
 
    PredictRequest (schemas.py) expects raw_h/raw_l as plain JSON arrays
    of floats -- np.ndarray isn't JSON-serializable, so .tolist() is
    required here (this upcasts float32 -> Python float/float64, which
    is fine: it's just the wire representation, not what the model
    computes on -- service.py re-casts to float32 before scaling).
    """
    payload = {
        "raw_h": window["raw_h"].tolist(),
        "raw_l": window["raw_l"].tolist(),
    }
    resp = requests.post(f"{api_url}/v1/predict", json=payload, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    result["ground_truth"] = window["drone_type"]
    result["correct"] = result["predicted_class"] == window["drone_type"]
    return result
 
 
def replay_over_http(stream: Iterator[dict], api_url: str, timeout: float = 5.0) -> Iterator[dict]:
    """
    Wraps a replay() stream, POSTing each window to the API as it's
    emitted and yielding the prediction result (with ground truth
    attached) instead of the raw window. A per-window network/HTTP
    failure is logged and skipped rather than killing the whole
    scenario run -- one dropped request during a long monitored replay
    shouldn't take down the client loop.
    """
    for window in stream:
        try:
            yield send_window(window, api_url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            print(f"  [!] request failed for rec={window['recording_id']} pos={window['seg_within_file']}: {e}")
 


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
        default="background:5,bebop:3,background:5,phantom:3,background:5,ar:3",
        help=(
            "Comma-separated scenario spec. Each token is either a bare "
            "class name (one occurrence) or 'class:count' (N consecutive "
            "occurrences), e.g. 'background:20000,ar:10000,background:5000'"
        ),
    )
    parser.add_argument("--speed", type=float, default=20.0, help="Playback speed multiplier (1.0 = real time)")
    parser.add_argument("--list-classes", action="store_true", help="Print available classes + pool sizes for this split and exit")
    parser.add_argument(
        "--no-reshuffle",
        action="store_true",
        help="Disable reshuffle-on-exhaust: reuse recordings in the same fixed order every lap instead of a fresh shuffle",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help=(
            "If set, POST each window to '<api-url>/v1/predict' instead of just "
            "printing shapes -- e.g. --api-url http://localhost:8000. Runs a "
            "/health check first and reports running accuracy vs. ground truth."
        ),
    )
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP request timeout in seconds (only used with --api-url)")

    args = parser.parse_args()

    h5_path = Path(args.h5)
    split_file = Path(args.splits)

    if args.list_classes:
        allowed_ids = _load_split_ids(split_file, args.split)
        idx = _build_recording_index(h5_path, allowed_ids)
        stats = class_recording_stats(idx)
        print(f"Classes available in split '{args.split}':")
        for cls, s in stats.items():
            avg_clips = s["total_clips"] / s["recordings"]
            print(
                f"  {cls:<12} : {s['recordings']:5d} distinct recordings, "
                f"{s['total_clips']:6d} total clips, {avg_clips:.1f} clips/recording avg"
            )
        return

    scenario = parse_scenario_spec(args.scenario) if args.mode == "scenario" else None
    if scenario is not None:
        print(f"Expanded scenario: {len(scenario)} windows total ({dict((c, scenario.count(c)) for c in set(scenario))})\n")

    stream = replay(
        h5_path=h5_path,
        split_file=split_file,
        split=args.split,
        mode=args.mode,
        scenario=scenario,
        speed_factor=args.speed,
        reshuffle_on_exhaust=not args.no_reshuffle,
    )

    print(f"Replaying split='{args.split}' mode='{args.mode}' speed={args.speed}x ...\n")

    if args.api_url:
        check_api_health(args.api_url, timeout=args.timeout)
        print(f"API at {args.api_url} is healthy. Streaming predictions...\n")
 
        n_seen, n_correct = 0, 0
        for i, result in enumerate(replay_over_http(stream, args.api_url, timeout=args.timeout)):
            n_seen += 1
            n_correct += result["correct"]
            mark = "OK " if result["correct"] else "ERR"
            print(
                f"[{i:04d}] {mark} truth={result['ground_truth']:<12} "
                f"pred={result['predicted_class']:<12} "
                f"conf={result['confidence']:.3f} "
                f"inference_ms={result['inference_time_ms']:.2f} "
                f"running_acc={n_correct / n_seen:.3f}"
            )
    else:
        for i, window in enumerate(stream):
            print(
                f"[{i:04d}] rec={window['recording_id']:<20} "
                f"pos={window['seg_within_file']:<3} "
                f"truth={window['drone_type']:<12} "
                f"raw_h={window['raw_h'].shape} raw_l={window['raw_l'].shape}"
            )
 


if __name__ == "__main__":
    main() 