"""
examine_before_api.py — pre-deployment inspection for the edge API artifacts

Purpose:
    Validate the files that the API will depend on before wiring up inference:
      - models/model.onnx
      - models/scaler.json
      - the processed feature files used to build requests

This script is intentionally lightweight and should be run before implementing
FastAPI/edge inference code. It reuses the same conventions as src/eval_onnx.py
and src/train.py.
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(f"onnxruntime is required to inspect the ONNX model: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(f"Could not find project root (looking for {marker})")


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(root_dir: Path) -> dict:
    processed = Path(os.getenv("FEATURE_OUT_FILE") or root_dir / "data/processed")
    model_dir = Path(os.getenv("MODEL_DIR") or root_dir / "models")
    metrics_dir = Path(os.getenv("METRICS_DIR") or root_dir / "metrics")
    return dict(
        h_scalars=processed / "H_scalars.npz",
        l_scalars=processed / "L_scalars.npz",
        model_onnx=model_dir / "model.onnx",
        scaler=model_dir / "scaler.json",
        scores=metrics_dir / "scores.json",
    )


def _load_sample_vectors(h_path: Path, l_path: Path) -> tuple[np.ndarray, np.ndarray]:
    h = np.load(h_path, allow_pickle=False)
    l = np.load(l_path, allow_pickle=False)
    X = np.hstack([h["X_scalar"], l["X_scalar"]]).astype(np.float32)
    y = h["y"].astype(np.int64)
    return X, y


def _apply_scaler(X: np.ndarray, scaler_path: Path) -> np.ndarray:
    scaler = json.loads(scaler_path.read_text())
    mean = np.array(scaler["mean"], dtype=np.float32)
    std = np.array(scaler["std"], dtype=np.float32)
    return (X - mean) / std


def inspect_scaler(scaler_path: Path) -> None:
    print("\n[1/4] Inspecting scaler.json")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")

    data = json.loads(scaler_path.read_text())
    print(f"  file: {scaler_path}")
    print(f"  keys: {sorted(data.keys())}")

    if "mean" not in data or "std" not in data:
        raise ValueError("scaler.json must contain 'mean' and 'std' arrays")

    mean = np.array(data["mean"], dtype=np.float32)
    std = np.array(data["std"], dtype=np.float32)

    if mean.shape != std.shape:
        raise ValueError(f"mean/std shape mismatch: {mean.shape} vs {std.shape}")

    print(f"  feature_count: {mean.shape[0]}")
    assert mean.shape == std.shape
    print(f"  mean and std shape: {mean.shape}")
    print(f"  mean_sample: {mean[:5].tolist()}")
    print(f"  std_sample:  {std[:5].tolist()}")

    if mean.shape[0] != 14:
        print("  warning: expected 14 scalar features for this model, got {mean.shape[0]}")


def inspect_feature_files(h_path: Path, l_path: Path) -> np.ndarray:
    print("\n[2/4] Inspecting processed feature files")
    if not h_path.exists() or not l_path.exists():
        raise FileNotFoundError(f"Missing processed feature files: {h_path} / {l_path}")

    X, y = _load_sample_vectors(h_path, l_path)
    print(f"  H file: {h_path}")
    print(f"  L file: {l_path}")
    print(f"  samples: {X.shape[0]}")
    print(f"  features_per_sample: {X.shape[1]}")
    print(f"  labels_shape: {y.shape}")
    print(f"  first_row: {X[0].tolist()}")

    if X.shape[1] != 14:
        raise ValueError(f"Expected 14 scalar features but got {X.shape[1]}")

    return X


def inspect_onnx_model(model_path: Path, sample_features: np.ndarray, scaler_path: Path) -> None:
    print("\n[3/4] Inspecting ONNX model")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {model_path}")

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print(f"  model: {model_path}")
    print(f"  providers: {sess.get_providers()}")

    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    print(f"  input_count: {len(inputs)}")
    print(f"  output_count: {len(outputs)}")

    input_info = inputs[0]
    output_info = outputs[0]
    print(f"  input_name: {input_info.name}")
    print(f"  input_shape: {input_info.shape}")
    print(f"  input_type: {input_info.type}")
    print(f"  output_name: {output_info.name}")
    print(f"  output_shape: {output_info.shape}")
    print(f"  output_type: {output_info.type}")

    # Apply the same preprocessing that the API will use.
    X_sample = _apply_scaler(sample_features[:1], scaler_path)
    X_sample = X_sample.reshape(1, 1, X_sample.shape[1])
    print(f"  sample_input_shape: {X_sample.shape}")

    logits = sess.run(None, {input_info.name: X_sample})[0]
    pred = int(np.argmax(logits, axis=1)[0])
    print(f"  inference_output_shape: {logits.shape}")
    print(f"  predicted_class_index: {pred}")
    print(f"  logits_sample: {logits[0].tolist()}")


def inspect_runtime_contract(root_dir: Path, cfg: dict, paths: dict) -> None:
    print("\n[4/4] Runtime contract summary")
    label_map = cfg["data_aggregator"]["label_map"]
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    print(f"  label_names: {label_names}")
    print(f"  expected input shape for API: (batch, 1, 14) after scaling")
    print(f"  expected output shape for API: (batch, {len(label_names)})")
    print(f"  model artifact: {paths['model_onnx']}")
    print(f"  scaler artifact: {paths['scaler']}")
    print("\nNext step: copy this preprocessing contract into your FastAPI request handler.")


def main() -> None:
    load_dotenv()

    root_dir = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    cfg = _load_config(root_dir / "config/params.yaml")
    paths = _resolve_paths(root_dir)

    print("DroneRF pre-API inspection")
    print(f"project_root: {root_dir}")

    inspect_scaler(paths["scaler"])
    sample_features = inspect_feature_files(paths["h_scalars"], paths["l_scalars"])
    inspect_onnx_model(paths["model_onnx"], sample_features, paths["scaler"])
    inspect_runtime_contract(root_dir, cfg, paths)


if __name__ == "__main__":
    main()
