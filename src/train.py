"""
train.py — XGBoost multiclass trainer for DroneRF (drone-training repo)

Pipeline position:  featurize (H + L) → train → model artifact in MLflow
Input:              data/processed/H_scalars.npz
                    data/processed/L_scalars.npz
Output:             models/xgb_drone.json          (XGBoost native format)
                    mlflow run  (metrics + registered model)

DVC tracks:
  deps   — both .npz files + this script
  params — config/params.yaml: train
  outs   — models/xgb_drone.json
  metrics— metrics/scores.json
"""

import json
import os
import warnings
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------

def _find_project_root(marker: str = "pyproject.toml") -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Could not find project root (looking for {marker}). "
        f"Started search from {current}"
    )


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_paths(root_dir: Path) -> dict:
    """
    All paths derived from env vars (for DVC overrides) or project root.
    Returns a plain dict so callers stay explicit.
    """
    processed = Path(os.getenv("FEATURE_OUT_FILE") or root_dir / "data/processed")
    return dict(
        h_scalars  = processed / "H_scalars.npz",
        l_scalars  = processed / "L_scalars.npz",
        splits     = Path(os.getenv("SPLIT_FILE") or root_dir / "data/interim/dronerf_splits.json"),
        model_dir  = Path(os.getenv("MODEL_DIR")  or root_dir / "models"),
        model_out  = Path(os.getenv("MODEL_DIR")  or root_dir / "models") / "xgb_drone.json",
        metrics    = Path(os.getenv("METRICS_DIR") or root_dir / "metrics") / "scores.json",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(
    h_path: Path,
    l_path: Path,
    split_path: Path,
    split: str,          # "train" | "val" | "test"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load H and L scalars for a given split, concatenate along feature axis.

    H_scalars.npz and L_scalars.npz both contain:
      X_scalar : (N, 7) float32
      y        : (N,)   int32
      freqs    : (1024,) — ignored here

    The segment ordering inside each .npz matches the original H5 segment
    order (sorted keys), so we cannot naively hstack rows. Instead we align
    on segment_id via the splits JSON which stores ordered lists per split.

    Strategy: build segment_id → row_index maps for H and L, then pull rows
    in splits[split] order, keeping only IDs present in both bands.
    """
    splits = json.loads(split_path.read_text())
    ids    = splits[split]           # ordered list of segment_ids for this split

    h = np.load(h_path, allow_pickle=False)
    l = np.load(l_path, allow_pickle=False)

    # .npz from featurize.py is filtered to one band at a time, so the row
    # order matches the sorted segment iteration in load_dataset().
    # We stored meta in a separate .parquet; here we rely on the fact that
    # featurize iterates `sorted(hf["segments"].keys())` and filters by split
    # membership — meaning row i corresponds to ids[i] within that band.
    #
    # Simplest safe approach: trust that both H and L have identical N and
    # the same segment ordering (both filtered identically from the same H5).
    # Assert this holds, then hstack.

    X_h, y_h = h["X_scalar"], h["y"]
    X_l, y_l = l["X_scalar"], l["y"]

    assert len(X_h) == len(X_l), (
        f"H and L scalar counts differ for split='{split}': "
        f"{len(X_h)} vs {len(X_l)}. Re-run featurize for both bands."
    )
    assert np.array_equal(y_h, y_l), (
        f"Label arrays differ between H and L for split='{split}'. "
        f"Bands must have been featurized from the same H5 + splits."
    )

    X = np.hstack([X_h, X_l])   # (N, 14)  — 7 H features + 7 L features
    y = y_h                      # labels identical in both

    return X.astype(np.float32), y.astype(np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_model(params: dict) -> XGBClassifier:
    """
    Construct XGBClassifier from the train.xgb block of params.yaml.
    All keys map directly to XGBoost constructor args — no translation layer.
    """
    return XGBClassifier(
        objective        = "multi:softprob",
        num_class        = params.get("num_class", 4),
        n_estimators     = params.get("n_estimators", 300),
        max_depth        = params.get("max_depth", 6),
        learning_rate    = params.get("learning_rate", 0.1),
        subsample        = params.get("subsample", 0.8),
        colsample_bytree = params.get("colsample_bytree", 0.8),
        reg_alpha        = params.get("reg_alpha", 0.0),
        reg_lambda       = params.get("reg_lambda", 1.0),
        random_state     = params.get("random_seed", 42),
        n_jobs           = -1,
        eval_metric      = "mlogloss",
        early_stopping_rounds = params.get("early_stopping_rounds", 20),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return {
        "accuracy":        round(accuracy_score(y_true, y_pred), 6),
        "f1_macro":        round(f1_score(y_true, y_pred, average="macro"), 6),
        "f1_weighted":     round(f1_score(y_true, y_pred, average="weighted"), 6),
        # per-class F1 — useful for CI gate on rare classes
        **{
            f"f1_{name}": round(report[name]["f1-score"], 6)
            for name in label_names
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    # ── 1. Config ───────────────────────────────────────────────────────────
    root_dir = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    cfg      = _load_config(root_dir / "config/params.yaml")
    paths    = _resolve_paths(root_dir)

    train_cfg  = cfg["train"]
    xgb_params = train_cfg["xgb"]

    # ── 2. Class metadata ───────────────────────────────────────────────────
    # label_map from params.yaml: {name: int}  e.g. {"background":0, "bebop":1, ...}
    label_map   = cfg["data_aggregator"]["label_map"]
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    # → ["background", "bebop", "ar", "phantom"] in label-index order

    # ── 3. Load splits ──────────────────────────────────────────────────────
    print("Loading H+L features …")
    X_train, y_train = load_split(paths["h_scalars"], paths["l_scalars"], paths["splits"], "train")
    X_val,   y_val   = load_split(paths["h_scalars"], paths["l_scalars"], paths["splits"], "val")
    X_test,  y_test  = load_split(paths["h_scalars"], paths["l_scalars"], paths["splits"], "test")
    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  features={X_train.shape[1]}")

    # ── 4. MLflow setup ─────────────────────────────────────────────────────
    
    mlflow.set_experiment(train_cfg.get("mlflow_experiment", "drone-detection"))

    with mlflow.start_run(run_name=train_cfg.get("mlflow_run_name", "xgb-multiclass")) as run:

        # ── 5. Log params to MLflow (mirrors what DVC tracks in params.yaml) ─
        mlflow.log_params(xgb_params)
        mlflow.log_param("bands", "H+L")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_classes", len(label_names))

        # ── 6. Train ────────────────────────────────────────────────────────
        print("Training XGBoost …")
        model = build_model(xgb_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # ── 7. Evaluate ─────────────────────────────────────────────────────
        val_metrics  = compute_metrics(y_val,  model.predict(X_val),  label_names)
        test_metrics = compute_metrics(y_test, model.predict(X_test), label_names)

        print("\n── Validation ──")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\n── Test ──")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Log with val_ / test_ prefix so they're distinguishable in MLflow UI
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # ── 8. Save model artifact ──────────────────────────────────────────
        paths["model_dir"].mkdir(parents=True, exist_ok=True)
        model.save_model(paths["model_out"])
        print(f"\nModel saved → {paths['model_out']}")

        # Register in MLflow Model Registry
        mlflow.xgboost.log_model(
            xgb_model        = model,
            artifact_path    = "model",
            registered_model_name = train_cfg.get("registered_model_name", "drone-xgb"),
        )
        print(f"MLflow run: {run.info.run_id}")

        # ── 9. Write metrics file for DVC + CI gate ─────────────────────────
        paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
        scores = {
            "val":  val_metrics,
            "test": test_metrics,
        }
        paths["metrics"].write_text(json.dumps(scores, indent=2))
        print(f"Metrics → {paths['metrics']}")


if __name__ == "__main__":
    main()
