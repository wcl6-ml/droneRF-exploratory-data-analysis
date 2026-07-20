"""
eval_onnx.py — ONNX model evaluation on workstation

Runs the exported model.onnx against the test split using onnxruntime
(CPU, no GPU needed). Writes metrics/onnx_scores.json for DVC tracking
and degradation comparison against metrics/scores.json (PyTorch baseline).

Pipeline position:  train → eval_onnx
Run on:             workstation (before touching edge device)

Data shape note:
    Scalars are loaded flat (N, 14) then reshaped to (N, 1, 14) to match
    the CNN1d ONNX model's expected input "signal_features".
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers  (copied pattern from train.py — no shared lib yet)
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
    processed   = Path(os.getenv("FEATURE_OUT_FILE") or root_dir / "data/processed")
    model_dir   = Path(os.getenv("MODEL_DIR")        or root_dir / "models")
    metrics_dir = Path(os.getenv("METRICS_DIR")      or root_dir / "metrics")
    return dict(
        h_scalar    = processed / "H_scalar.npz",
        l_scalar    = processed / "L_scalar.npz",
        h_psd  = processed / "H_psd.npz",
        l_psd  = processed / "L_psd.npz",
        model_onnx   = model_dir / "model.onnx",
        scaler       = model_dir / "scaler.json",
        scores       = metrics_dir / "scores.json",        # PyTorch baseline
        onnx_scores  = metrics_dir / "onnx_scores.json",  # this script's output
    )


def _load_test_split(h_path: Path, l_path: Path, data_used: str) -> tuple[np.ndarray, np.ndarray]:
    h = np.load(h_path, allow_pickle=False)
    l = np.load(l_path, allow_pickle=False)
    X = np.hstack([h[f"X_{data_used}"], l[f"X_{data_used}"]]).astype(np.float32)
    y = h["y"].astype(np.int64)
    return X, y


def _apply_scaler(X: np.ndarray, scaler_path: Path) -> np.ndarray:
    scaler = json.loads(scaler_path.read_text())
    mean   = np.array(scaler["mean"], dtype=np.float32)
    std    = np.array(scaler["std"],  dtype=np.float32)
    return (X - mean) / std


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        **{f"f1_{n}": round(report[n]["f1-score"], 6) for n in label_names},
    }


def _print_degradation(baseline: dict, onnx: dict) -> None:
    """
    Compare ONNX metrics against PyTorch baseline.
    Flags any metric that degrades by more than 1%.
    """
    print("\n── Degradation (ONNX vs PyTorch test) ──")
    print(f"  {'metric':<20} {'pytorch':>10} {'onnx':>10} {'delta':>10}")
    print(f"  {'-'*52}")

    warned = False
    for k in baseline:
        b = baseline[k]
        o = onnx.get(k, float("nan"))
        delta = o - b
        flag  = " ⚠" if delta < -0.01 else ""
        if flag:
            warned = True
        print(f"  {k:<20} {b:>10.4f} {o:>10.4f} {delta:>+10.4f}{flag}")

    if warned:
        print("\n  ⚠  Degradation > 1% detected — check quantisation / opset.")
    else:
        print("\n  ✓  ONNX parity within 1% on all metrics.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    root_dir    = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    cfg         = _load_config(root_dir / "config/params.yaml")
    train_cfg   = cfg["train"]
    data_used = train_cfg.get("data_used", "scalars")
    paths       = _resolve_paths(root_dir)

    label_map   = cfg["data_aggregator"]["label_map"]
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]

    # ── 1. Load + normalise test split ──────────────────────────────────────
    print(f"Loading test {data_used} …")
    X_test, y_test = _load_test_split(paths[f"h_{data_used}"], paths[f"l_{data_used}"], data_used)
    X_test = _apply_scaler(X_test, paths["scaler"])
    # Reshape flat (N, 14) → (N, 1, 14) to match CNN1d ONNX input shape
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"  test={len(X_test)}  shape={X_test.shape}")

    # ── 2. ONNX inference ───────────────────────────────────────────────────
    print(f"\nLoading ONNX model: {paths['model_onnx']}")
    sess = ort.InferenceSession(
        str(paths["model_onnx"]),
        providers=["CPUExecutionProvider"],   # CPU on workstation for parity check
    )
    input_name = sess.get_inputs()[0].name    # "signal_features"

    logits = sess.run(None, {input_name: X_test})[0]  # (N, 4)
    y_pred = logits.argmax(axis=1)

    # ── 3. Metrics ──────────────────────────────────────────────────────────
    onnx_metrics = _compute_metrics(y_test, y_pred, label_names)

    print("\n── ONNX test metrics ──")
    for k, v in onnx_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── 4. Degradation report ───────────────────────────────────────────────
    if paths["scores"].exists():
        baseline = json.loads(paths["scores"].read_text())["test"]
        _print_degradation(baseline, onnx_metrics)
    else:
        print("\n  scores.json not found — skipping degradation report.")
        print("  Run train.py first to generate the PyTorch baseline.")

    # ── 5. Write onnx_scores.json — DVC metrics ─────────────────────────────
    paths["onnx_scores"].parent.mkdir(parents=True, exist_ok=True)
    paths["onnx_scores"].write_text(json.dumps({"test": onnx_metrics}, indent=2))
    print(f"\n  ONNX scores → {paths['onnx_scores']}")


if __name__ == "__main__":
    main()
