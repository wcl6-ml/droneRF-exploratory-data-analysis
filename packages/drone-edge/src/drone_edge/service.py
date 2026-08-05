"""
service.py
==========
The compute core of the API. 

This module is part of the deployable edge artifact, alongside
predictor.py, model.onnx, scaler.json, and labels.json. Anything that
needs the dataset (label ordering, evaluation, integration testing)
lives outside this file -- see src/utils/generate_labels.py for how
labels.json gets produced.

Responsibilities:
  - load the ONNX model + scaler + label names exactly once, at process
    startup -- never per-request
  - stay fully testable by calling it directly, no server needed
  - stay a thin wrapper: main.py (the HTTP layer, next script) parses
    requests and calls this; nothing about HTTP leaks in here, and
    nothing about the dataset leaks in from here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.api.predictor import DroneRFPredictor


def load_label_names(labels_path: Path) -> list[str]:
    """
    Loads the class ordering from labels.json -- a small, static artifact
    generated once (offline, wherever the dataset lives) by
    src/utils/generate_labels.py from config/params.yaml's label_map.

    This file, not params.yaml and not the .h5, is what ships with the
    model on the edge. No dataset dependency here at all.
    """
    return json.loads(Path(labels_path).read_text())


class ModelService:
    """
    Thin, HTTP-agnostic, dataset-agnostic wrapper around DroneRFPredictor.

    Usage:
        service = ModelService(model_path, scaler_path, labels_path).load()
        result = service.predict(raw_h, raw_l)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        labels_path: Optional[str] = None,
    ):
        self._labels_path = Path(labels_path or "models/labels.json")
        self._predictor = DroneRFPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            label_names=None,  # set in load(), from labels.json
        )
        self._loaded = False

    def load(self) -> "ModelService":
        """Loads label names, ONNX session, and scaler once. Call at process startup."""
        self._predictor.label_names = load_label_names(self._labels_path)
        self._predictor.load()
        self._loaded = True
        return self

    @property
    def is_ready(self) -> bool:
        return self._loaded

    @property
    def classes(self) -> list[str]:
        return self._predictor.label_names

    def predict(self, raw_h, raw_l) -> dict:
        """
        raw_h / raw_l: flat interleaved I/Q sequences (list or ndarray).
        Returns predictor.predict()'s dict plus inference latency -- the
        same shape this service hands back through the API unchanged,
        so this is exactly what main.py will serialize as JSON next.
        """
        if not self._loaded:
            raise RuntimeError("ModelService.load() must be called before predict()")

        start = time.perf_counter()
        result = self._predictor.predict(
            np.asarray(raw_h, dtype=np.float32),
            np.asarray(raw_l, dtype=np.float32),
        )
        result["inference_time_ms"] = round((time.perf_counter() - start) * 1000, 3)
        return result


# ---------------------------------------------------------------------------
# Standalone self-check -- proves the service loads and runs, using a
# synthetic window. Deliberately does NOT touch the dataset or the
# simulator: this file has no business knowing either exists. Real
# correctness-against-ground-truth checking happens later, over HTTP,
# once main.py exists -- that's the honest integration test, matching
# how a real sensor's edge software would actually exercise this API.
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test ModelService with a synthetic window (no dataset).")
    parser.add_argument("--model", default="models/model.onnx")
    parser.add_argument("--scaler", default="models/scaler.json")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--segment-length", type=int, default=100_000, help="Must match config/params.yaml's segment_length")
    args = parser.parse_args()

    service = ModelService(
        model_path=args.model,
        scaler_path=args.scaler,
        labels_path=args.labels,
    ).load()

    print(f"Model loaded. Classes: {service.classes}\n")

    # Synthetic noise standing in for a real window -- this test only
    # proves the pipeline runs end to end (DSP -> scaling -> ONNX ->
    # softmax -> label lookup), not that predictions are meaningful.
    rng = np.random.default_rng(0)
    raw_h = (rng.standard_normal(args.segment_length) * 0.1).astype(np.float32)
    raw_l = (rng.standard_normal(args.segment_length) * 0.1).astype(np.float32)

    result = service.predict(raw_h, raw_l)
    print("Synthetic-window prediction (sanity check only, not accuracy):")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()