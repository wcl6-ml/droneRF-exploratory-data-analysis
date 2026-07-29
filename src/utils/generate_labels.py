"""
generate_labels.py
===================
Dev-time-only utility. Reads the single source of truth for class
ordering (config/params.yaml's data_aggregator.label_map, DVC-tracked)
and writes models/labels.json -- a tiny artifact that ships alongside
model.onnx and scaler.json.

Run this once whenever label_map changes, then commit the resulting
models/labels.json. service.py reads that file at startup and never
touches params.yaml or DVC directly -- keeping the deployed/edge code
free of any dependency on the dataset or the training config.

Usage:
    python src/utils/generate_labels.py
    python src/utils/generate_labels.py --params config/params.yaml --out models/labels.json
"""

import argparse
import json
from pathlib import Path

import yaml


def generate_labels(params_path: Path, out_path: Path) -> list[str]:
    cfg = yaml.safe_load(params_path.read_text())
    label_map = cfg["data_aggregator"]["label_map"]  # {"background": 0, "bebop": 1, ...}

    if set(label_map.values()) != set(range(len(label_map))):
        raise ValueError(f"label_map values are not a contiguous 0..N-1 range: {label_map}")

    # Index i must be the class whose label int is i -- this ordering is
    # exactly what the model's output logits are indexed by.
    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(label_names, indent=2))
    return label_names


def main():
    parser = argparse.ArgumentParser(description="Generate models/labels.json from config/params.yaml")
    parser.add_argument("--params", default="config/params.yaml")
    parser.add_argument("--out", default="models/labels.json")
    args = parser.parse_args()

    label_names = generate_labels(Path(args.params), Path(args.out))
    print(f"Wrote {args.out}: {list(enumerate(label_names))}")
    print(label_names)

if __name__ == "__main__":
    main()