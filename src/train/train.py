"""
train.py — CNN1d trainer for DroneRF scalar features

Pipeline position:  featurize (H + L) → train → model.onnx + metrics/scores.json
Input:              data/processed/H_scalars.npz
                    data/processed/L_scalars.npz
                    data/processed/H_psd.npz
                    data/processed/L_psd.npz
Output:             models/model.onnx          (deployment artifact, DVC tracked)
                    models/model.pt            (checkpoint, DVC tracked)
                    models/scaler.json         (normalisation stats, DVC tracked)
                    metrics/scores.json        (DVC metrics)

MLflow logs:        params, per-epoch loss/acc, val/test metrics, model artifact

Data shape note:
    Raw scalars are (N, 14) float32. They are reshaped to (N, 1, 14) before
    being fed to the CNN — treating the 14 features as a 1-channel sequence
    of length 14. The scaler is still fit/applied on the flat (N, 14) form.
"""

import json
import os
import warnings
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import yaml
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

from models.model import DroneCNN1dClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config / path helpers  (same pattern as featurize.py)
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
        h_scalar  = processed / "H_scalar.npz",
        l_scalar  = processed / "L_scalar.npz",
        h_psd  = processed / "H_psd.npz",
        l_psd  = processed / "L_psd.npz",
        splits     = Path(os.getenv("SPLIT_FILE") or root_dir / "data/interim/dronerf_splits.json"),
        model_dir  = model_dir,
        model_pt   = model_dir / "model.pt",
        model_onnx = model_dir / "model.onnx",
        scaler     = model_dir / "scaler.json",
        scores     = metrics_dir / "scores.json",
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(
    h_path: Path,
    l_path: Path,
    split:  str,
    data_used: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load H and L scalars, hstack to (N, 14), return (X, y).
    Relies on both .npz files having identical row ordering and labels —
    guaranteed when featurize runs on the same H5 + splits file.
    """
    h = np.load(h_path, allow_pickle=False)
    l = np.load(l_path, allow_pickle=False)

    X_h, y_h = h[f"X_{data_used}"], h["y"]
    X_l, y_l = l[f"X_{data_used}"], l["y"]

    assert len(X_h) == len(X_l), (
        f"H/L row count mismatch for split='{split}': {len(X_h)} vs {len(X_l)}"
    )
    assert np.array_equal(y_h, y_l), (
        f"H/L label mismatch for split='{split}' — re-run featurize for both bands."
    )

    X = np.hstack([X_h, X_l]).astype(np.float32)   # (N, 14)
    y = y_h.astype(np.int64)
    return X, y


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # Reshape flat (N, 14) → (N, 1, 14) for Conv1d: 1 channel, sequence length 14
    X_3d = X.reshape(X.shape[0], 1, X.shape[1])
    ds = TensorDataset(torch.from_numpy(X_3d), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


# ---------------------------------------------------------------------------
# Normalisation  (fit on train only, apply to val/test)
# ---------------------------------------------------------------------------

def _fit_scaler(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return mean, std


def _apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    model:       nn.Module,
    loader:      DataLoader,
    device:      torch.device,
    label_names: list[str],
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)

    return {
        "accuracy":    round(float(accuracy_score(y_true, y_pred)), 6),
        "f1_macro":    round(float(f1_score(y_true, y_pred, average="macro")), 6),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted")), 6),
        **{f"f1_{n}": round(report[n]["f1-score"], 6) for n in label_names},
    }


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def _export_onnx(model: nn.Module, onnx_path: Path, seq_len: int) -> None:
    model.eval().cpu()
    dummy = torch.zeros(1, 1, seq_len)   # (batch=1, channels=1, length=seq_len)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names  = ["signal_features"],
        output_names = ["logits"],
        dynamic_axes = {
            "signal_features": {0: "batch_size"},
            "logits":          {0: "batch_size"},
        },
        opset_version = 17,
    )
    print(f"  ONNX → {onnx_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    # ── 1. Config ───────────────────────────────────────────────────────────
    root_dir    = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    cfg         = _load_config(root_dir / "config/params.yaml")
    paths       = _resolve_paths(root_dir)
    train_cfg   = cfg["train"]
    data_used = train_cfg.get("data_used", "scalars")
    cnn_params  = train_cfg["cnn1d"]

    label_map   = cfg["data_aggregator"]["label_map"]
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    num_classes = len(label_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 2. Data ─────────────────────────────────────────────────────────────
    print(f"Loading {data_used} …")
    X_train, y_train = _load_split(paths[f"h_{data_used}"], paths[f"l_{data_used}"], "train", data_used)
    X_val,   y_val   = _load_split(paths[f"h_{data_used}"], paths[f"l_{data_used}"], "val", data_used)
    X_test,  y_test  = _load_split(paths[f"h_{data_used}"], paths[f"l_{data_used}"], "test", data_used)

    mean, std = _fit_scaler(X_train)
    X_train   = _apply_scaler(X_train, mean, std)
    X_val     = _apply_scaler(X_val,   mean, std)
    X_test    = _apply_scaler(X_test,  mean, std)

    batch_size   = cnn_params.get("batch_size", 256)
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader   = _make_loader(X_val,   y_val,   batch_size, shuffle=False)
    test_loader  = _make_loader(X_test,  y_test,  batch_size, shuffle=False)

    print(f"  train={len(X_train)}  val={len(X_val)}  test={len(X_test)}  features={X_train.shape[1]}")

    # ── 3. Model ────────────────────────────────────────────────────────────
    model = DroneCNN1dClassifier(
        in_channels = 1,
        seq_len     = X_train.shape[1],          # 14
        num_filters = cnn_params.get("num_filters", 64),
        kernel_size = cnn_params.get("kernel_size", 3),
        num_classes = num_classes,
        dropout     = cnn_params.get("dropout", 0.3),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cnn_params.get("lr", 1e-3),
        weight_decay = cnn_params.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cnn_params.get("epochs", 50)
    )

    # ── 4. MLflow ───────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(root_dir / "mlruns")
    mlflow.set_experiment(train_cfg.get("mlflow_experiment", "drone-detection"))

    with mlflow.start_run(run_name=train_cfg.get("mlflow_run_name", "cnn1d-scalars")) as run:
        mlflow.log_params(cnn_params)
        mlflow.log_param("bands", "H+L")
        mlflow.log_param("in_channels", 1)
        mlflow.log_param("seq_len", X_train.shape[1])
        mlflow.log_param("num_classes", num_classes)

        # ── 5. Training loop ─────────────────────────────────────────────
        epochs      = cnn_params.get("epochs", 50)
        best_val_f1 = 0.0
        best_epoch  = 0

        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, correct, total = 0.0, 0, 0

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(y_batch)
                correct    += (logits.argmax(1) == y_batch).sum().item()
                total      += len(y_batch)

            scheduler.step()

            train_loss = total_loss / total
            train_acc  = correct / total
            val_metrics = _evaluate(model, val_loader, device, label_names)

            mlflow.log_metrics({
                "train_loss": round(train_loss, 6),
                "train_acc":  round(train_acc, 6),
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  [{epoch:03d}/{epochs}] "
                    f"loss={train_loss:.4f}  "
                    f"train_acc={train_acc:.4f}  "
                    f"val_f1={val_metrics['f1_macro']:.4f}"
                )

            # Save best checkpoint by val f1_macro
            if val_metrics["f1_macro"] > best_val_f1:
                best_val_f1 = val_metrics["f1_macro"]
                best_epoch  = epoch
                paths["model_dir"].mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), paths["model_pt"])

        print(f"\nBest val f1_macro={best_val_f1:.4f} at epoch {best_epoch}")

        # ── 6. Evaluate best checkpoint ──────────────────────────────────
        model.load_state_dict(torch.load(paths["model_pt"], map_location=device))
        val_metrics  = _evaluate(model, val_loader,  device, label_names)
        test_metrics = _evaluate(model, test_loader, device, label_names)

        mlflow.log_metrics({f"best_val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print("\n── Val (best checkpoint) ──")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        print("\n── Test ──")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")

        # ── 7. ONNX export ───────────────────────────────────────────────
        _export_onnx(model, paths["model_onnx"], seq_len=X_train.shape[1])
        mlflow.log_artifact(str(paths["model_onnx"]), artifact_path="onnx")
        mlflow.log_artifact(str(paths["model_pt"]),   artifact_path="checkpoint")

        # ── 8. Save scaler — required at inference time ──────────────────
        paths["scaler"].write_text(json.dumps({
            "mean": mean.tolist(),
            "std":  std.tolist(),
        }))
        mlflow.log_artifact(str(paths["scaler"]), artifact_path="scaler")
        print(f"  Scaler → {paths['scaler']}")

        # ── 9. metrics/scores.json — DVC metrics + CI gate ───────────────
        paths["scores"].parent.mkdir(parents=True, exist_ok=True)
        scores = {"val": val_metrics, "test": test_metrics}
        paths["scores"].write_text(json.dumps(scores, indent=2))
        mlflow.log_artifact(str(paths["scores"]))
        print(f"  Scores → {paths['scores']}")
        print(f"\nMLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()
