# -*- coding: utf-8 -*-
"""
eda_pipeline.py
===============
EDA + DSP pipeline for the DroneRF HDF5 archive.

Pipeline:
    H5 segment (interleaved IQ, float32)
        → deinterleave → complex64
        → Welch PSD  (two-sided, dB)          ← single DSP method
        → 7 scalar spectral features
        → 5 EDA plots  (reports/figures/)
        → XGBoost classifier
        → confusion matrix + results.csv

Design decisions documented inline. Key ones:
  - One DSP method: Welch PSD on complex IQ (two-sided spectrum).
  - XGBoost on 7 scalar features only — interpretable baseline.
  - H-band default; L-band via --band L.
  - Splits loaded from splits.json, never re-generated here.
  - No segment-length flexibility: 1e5 is the H5 atom. Longer/shorter
    windows are a loader concern, not a DSP concern — add later.

Run:
    python eda_pipeline.py                  # full run, H-band
    python eda_pipeline.py --band L         # L-band
    python eda_pipeline.py --no-plots       # skip EDA, just train
    python eda_pipeline.py --max-segs 400   # smoke-test on subset
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — safe on servers without a display
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

root_dir   = Path(__file__).resolve().parent.parent.parent
H5_FILE    = root_dir / "data/interim/dronerf.h5"
SPLIT_FILE = root_dir / "data/interim/dronerf_splits.json"
FIG_DIR    = root_dir / "reports/figures"
RES_FILE   = root_dir / "reports/results.csv"

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# DSP constants
# ---------------------------------------------------------------------------

FS_H   = 40e6    # H-band sample rate (Hz)
FS_L   = 10e6    # L-band sample rate (Hz)

# Welch PSD
NPERSEG  = 1024
NOVERLAP = 768     # 75% overlap
NFFT     = 1024
WINDOW   = "hann"

# STFT spectrogram (for the EDA plot only — not used in feature extraction)
# Smaller nperseg than Welch: we want time resolution to see
# blade-rate AM modulation (~100–400 Hz), not just spectral shape.
STFT_NPERSEG  = 256
STFT_NOVERLAP = 192   # 75% overlap

N_FREQ_BINS = NFFT // 2 + 1   # 513 for two-sided after fftshift

# ---------------------------------------------------------------------------
# Labels / colours
# ---------------------------------------------------------------------------

CLASS_NAMES  = ["background", "bebop", "ar", "phantom"]
CLASS_COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
LABEL_MAP    = {"background": 0, "bebop": 1, "ar": 2, "phantom": 3}
INV_LABEL    = {v: k for k, v in LABEL_MAP.items()}

SCALAR_NAMES = [
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
    "spectral_flatness",  "spectral_entropy",   "peak_freq", "snr_db",
]
#%%

# ---------------------------------------------------------------------------
# 1.  IQ helpers
# ---------------------------------------------------------------------------

def deinterleave(x: np.ndarray) -> np.ndarray:
    """
    Interleaved float32 [I0, Q0, I1, Q1, ...] → complex64, length N/2.

    This is the only IQ interpretation used. The H5 attr
    signal_format="interleaved_iq" confirms it is correct for DroneRF.
    """
    assert x.ndim == 1 and len(x) % 2 == 0, \
        f"Expected 1D even-length array, got shape {x.shape}"
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64)


# ---------------------------------------------------------------------------
# 2.  Welch PSD  (primary DSP method)
# ---------------------------------------------------------------------------

def welch_psd(
    iq: np.ndarray,
    fs: float,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
    nfft: int     = NFFT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-sided Welch PSD of a complex IQ signal.

    Returns: (freqs_hz, psd_db), both shape (nfft,) after fftshift.

    Why two-sided on complex (not rfft on real)?
      After deinterleave() the signal is complex baseband. Negative
      frequencies represent the lower sideband around the RF carrier.
      return_onesided=False preserves this information.
      fftshift moves DC to the centre so plots read -fs/2 → +fs/2.

    Frequency resolution: fs / nfft
      H-band: 40e6 / 1024 ≈ 39 kHz/bin
      L-band: 10e6 / 1024 ≈ 10 kHz/bin
    """
    freqs, psd = scipy_signal.welch(
        iq, fs=fs,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        scaling="density",
    )
    freqs  = np.fft.fftshift(freqs)
    psd_db = 10 * np.log10(np.fft.fftshift(np.abs(psd)) + 1e-12)
    return freqs.astype(np.float32), psd_db.astype(np.float32)


# ---------------------------------------------------------------------------
# 3.  STFT spectrogram  (EDA only — not used for features)
# ---------------------------------------------------------------------------

def stft_spectrogram(
    iq: np.ndarray,
    fs: float,
    nperseg: int  = STFT_NPERSEG,
    noverlap: int = STFT_NOVERLAP,
    nfft: int     = NFFT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    STFT magnitude spectrogram of a complex IQ signal.

    Returns: (freqs_hz, times_s, Sxx_db)
      freqs_hz : (nfft,)          — two-sided, fftshifted
      times_s  : (n_frames,)
      Sxx_db   : (nfft, n_frames) — magnitude in dB

    Smaller nperseg than Welch (256 vs 1024) gives finer time resolution:
      time step  = (nperseg - noverlap) / fs = 64 / 40e6 ≈ 1.6 µs/frame
      freq res   = fs / nfft = 40e6 / 1024 ≈ 39 kHz/bin
    This lets the spectrogram show blade-rate AM modulation in time,
    which the mean PSD collapses away.
    """
    freqs, times, Zxx = scipy_signal.stft(
        iq, fs=fs,
        window=WINDOW,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
    )
    freqs  = np.fft.fftshift(freqs)
    Zxx    = np.fft.fftshift(Zxx, axes=0)
    Sxx_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    return freqs.astype(np.float32), times.astype(np.float32), Sxx_db.astype(np.float32)


# ---------------------------------------------------------------------------
# 4.  Scalar features  (XGBoost input)
# ---------------------------------------------------------------------------

def spectral_scalars(freqs: np.ndarray, psd_db: np.ndarray) -> np.ndarray:
    """
    7 scalar features from a Welch PSD → shape (7,) float32.

    Returned in SCALAR_NAMES order so feature importance plots align.

    spectral_centroid   : frequency centre of mass (Hz).
    spectral_bandwidth  : weighted std dev around centroid (Hz).
    spectral_rolloff_85 : freq below which 85% of power sits (Hz).
    spectral_flatness   : geometric/arithmetic mean ratio.
                          0 = pure tone, 1 = white noise.
    spectral_entropy    : Shannon entropy of normalised PSD.
                          High = complex/broadband, Low = tonal.
    peak_freq           : frequency of max PSD bin (Hz).
    snr_db              : peak bin power – median power (dB proxy).
    """
    psd_lin  = 10 ** (psd_db / 10.0)
    psd_norm = psd_lin / (psd_lin.sum() + 1e-12)

    centroid  = float(np.sum(freqs * psd_norm))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm)))
    cumsum    = np.cumsum(psd_norm)
    rolloff   = float(freqs[np.searchsorted(cumsum, 0.85)])
    flatness  = float(
        np.exp(np.mean(np.log(psd_lin + 1e-12))) / (np.mean(psd_lin) + 1e-12)
    )
    entropy   = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
    peak_idx  = int(np.argmax(psd_lin))
    peak_freq = float(freqs[peak_idx])
    snr_db    = float(psd_db[peak_idx] - np.median(psd_db))

    return np.array(
        [centroid, bandwidth, rolloff, flatness, entropy, peak_freq, snr_db],
        dtype=np.float32,
    )

#%%
# ---------------------------------------------------------------------------
# 5.  Data loader
# ---------------------------------------------------------------------------

def load_dataset(
    h5_path:    Path,
    split_path: Path,
    band:       str       = "H",
    max_segs:   int|None  = None,
) -> dict:
    """
    Load all segments for a given band, run DSP, return arrays.

    Returns dict:
      X_scalar : (N, 7)    float32  — XGBoost features
      X_psd    : (N, 1024) float32  — full PSD (kept for EDA plots)
      y        : (N,)      int32    — class labels
      freqs    : (1024,)   float32  — shared frequency axis (Hz)
      meta     : list[dict]          — segment provenance
      splits   : dict                — {"train": [...], "val": [...], "test": [...]}
    """
    splits  = json.loads(split_path.read_text())
    all_ids = set(splits["train"] + splits["val"] + splits["test"])
    fs      = FS_H if band == "H" else FS_L

    X_scalar_list, X_psd_list, y_list, meta_list = [], [], [], []
    freqs_ref = None
    n_loaded  = 0

    print(f"\nLoading band={band}  fs={fs/1e6:.0f} MHz ...")
    t0 = time.perf_counter()

    with h5py.File(h5_path, "r") as hf:
        for key in sorted(hf["segments"].keys()):
            if key not in all_ids:
                continue
            seg = hf[f"segments/{key}"]
            if seg.attrs["band"] != band:
                continue

            raw              = seg["signal"][:]
            iq               = deinterleave(raw)
            freqs, psd_db    = welch_psd(iq, fs=fs)
            scalars          = spectral_scalars(freqs, psd_db)

            if freqs_ref is None:
                freqs_ref = freqs

            X_scalar_list.append(scalars)
            X_psd_list.append(psd_db)
            y_list.append(int(seg.attrs["label"]))
            meta_list.append({
                "segment_id":   key,
                "drone_type":   str(seg.attrs["drone_type"]),
                "label":        int(seg.attrs["label"]),
                "band":         str(seg.attrs["band"]),
                "recording_id": str(seg.attrs["recording_id"]),
            })

            n_loaded += 1
            if max_segs and n_loaded >= max_segs:
                print(f"  [--max-segs] capped at {max_segs}")
                break

    elapsed = time.perf_counter() - t0
    print(f"  {n_loaded} segments  |  "
          f"{elapsed:.1f}s total  |  "
          f"{elapsed / max(n_loaded, 1) * 1000:.1f} ms/seg")

    return dict(
        X_scalar = np.array(X_scalar_list, dtype=np.float32),
        X_psd    = np.array(X_psd_list,    dtype=np.float32),
        y        = np.array(y_list,         dtype=np.int32),
        freqs    = freqs_ref,
        meta     = meta_list,
        splits   = splits,
    )


def split_indices(meta: list[dict], splits: dict, name: str) -> np.ndarray:
    """Row indices into meta / X / y arrays for the named split."""
    id_set = set(splits[name])
    return np.array([i for i, m in enumerate(meta) if m["segment_id"] in id_set])


# ---------------------------------------------------------------------------
# 6.  EDA plots
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_class_balance(meta: list[dict], out_dir: Path) -> None:
    """Segment count per class — first check for class imbalance."""
    labels, counts = np.unique(
        [m["drone_type"] for m in meta], return_counts=True
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        labels, counts,
        color=[CLASS_COLORS[LABEL_MAP[c]] for c in labels],
        edgecolor="white", linewidth=0.8,
    )
    ax.bar_label(bars, padding=4, fontsize=10)
    ax.set_xlabel("Class")
    ax.set_ylabel("Segments")
    ax.set_title("Class Balance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "01_class_balance.png")


def plot_mean_psd_per_class(
    X_psd: np.ndarray, y: np.ndarray,
    freqs: np.ndarray, band: str,
    out_dir: Path,
) -> None:
    """
    Overlay mean ± 1 std PSD for each class.

    This is the most diagnostic EDA plot: if the four curves visually
    separate, the classification task is tractable with PSD features.
    If they overlap significantly, a richer representation (STFT/CNN)
    will be needed.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    freqs_mhz = freqs / 1e6

    for lbl, name in INV_LABEL.items():
        mask = y == lbl
        if not mask.any():
            continue
        mu  = X_psd[mask].mean(axis=0)
        sig = X_psd[mask].std(axis=0)
        c   = CLASS_COLORS[lbl]
        ax.plot(freqs_mhz, mu, label=name, color=c, lw=1.6)
        ax.fill_between(freqs_mhz, mu - sig, mu + sig, alpha=0.12, color=c)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title(f"Mean PSD per Class ± 1 std  |  Band {band}")
    ax.legend(framealpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / f"02_mean_psd_per_class_{band}.png")


def plot_example_spectrogram_per_class(
    h5_path: Path, meta: list[dict],
    band: str, out_dir: Path,
) -> None:
    """
    STFT spectrogram of one example segment per class.

    Time × frequency image. This complements the mean PSD by showing
    temporal structure — blade-rate AM appears as horizontal striping,
    carrier drift as tilted ridges.

    Layout: 2 × 2 grid, one subplot per class.
    Colour axis is clipped to [median-3σ, max] per segment to avoid
    a single hot pixel washing out the colour scale.
    """
    fs = FS_H if band == "H" else FS_L

    # One example segment per class (first encountered)
    examples: dict[str, str] = {}
    for m in meta:
        if m["drone_type"] not in examples:
            examples[m["drone_type"]] = m["segment_id"]
        if len(examples) == len(CLASS_NAMES):
            break

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    with h5py.File(h5_path, "r") as hf:
        for ax, name in zip(axes, CLASS_NAMES):
            if name not in examples:
                ax.set_visible(False)
                continue

            key    = examples[name]
            raw    = hf[f"segments/{key}/signal"][:]
            iq     = deinterleave(raw)
            f, t, S = stft_spectrogram(iq, fs=fs)

            # Robust colour limits
            vmin = float(np.median(S) - 3 * S.std())
            vmax = float(S.max())

            im = ax.pcolormesh(
                t * 1e3,          # seconds → milliseconds
                f  / 1e6,         # Hz → MHz
                S,
                shading="auto",
                cmap="inferno",
                vmin=vmin, vmax=vmax,
            )
            fig.colorbar(im, ax=ax, label="dB", pad=0.02)
            ax.set_title(f"{name}  (seg {key})", fontsize=10)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Frequency (MHz)")

    fig.suptitle(
        f"STFT Spectrogram — one example per class  |  Band {band}\n"
        f"nperseg={STFT_NPERSEG}, noverlap={STFT_NOVERLAP}, nfft={NFFT}",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir / f"03_spectrogram_per_class_{band}.png")


def plot_feature_boxplots(
    X_scalar: np.ndarray, y: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Boxplot of each scalar feature split by class.

    Read this before trusting model results:
      - Features where boxes don't overlap → strong discriminators.
      - Features where all boxes stack on top of each other → weak,
        may be dropped or replaced.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()

    for i, fname in enumerate(SCALAR_NAMES):
        ax = axes[i]
        data = [X_scalar[y == j, i] for j in range(len(CLASS_NAMES))]
        bp = ax.boxplot(
            data, labels=CLASS_NAMES,
            patch_artist=True, notch=False,
            medianprops={"color": "white", "lw": 2},
            flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        )
        for patch, color in zip(bp["boxes"], CLASS_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        ax.set_title(fname, fontsize=9)
        ax.tick_params(axis="x", labelsize=7, rotation=15)
        ax.spines[["top", "right"]].set_visible(False)

    axes[-1].set_visible(False)   # 7 features, 8 subplots
    fig.suptitle("Scalar Feature Distributions per Class", y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir / "04_feature_boxplots.png")


def plot_band_comparison(
    h5_path: Path, splits: dict, out_dir: Path,
) -> None:
    """
    H-band vs L-band mean PSD per class on a normalised frequency axis.

    Frequency axes differ (40 MHz vs 10 MHz) so we normalise to [0, 1]
    for visual comparison. The shape of each band's PSD tells you whether
    the two bands carry complementary or redundant information — key input
    for deciding whether band-fusion experiments are worth pursuing.
    """
    all_ids = set(splits["train"] + splits["val"] + splits["test"])
    psd_store: dict[tuple[str, str], list[np.ndarray]] = {}

    with h5py.File(h5_path, "r") as hf:
        for key in sorted(hf["segments"].keys()):
            if key not in all_ids:
                continue
            seg  = hf[f"segments/{key}"]
            band = str(seg.attrs["band"])
            dt   = str(seg.attrs["drone_type"])
            fs   = float(seg.attrs["fs_hz"])
            raw  = seg["signal"][:]
            iq   = deinterleave(raw)
            _, psd = welch_psd(iq, fs=fs)
            psd_store.setdefault((band, dt), []).append(psd)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()
    styles = {"H": ("-",  0.9), "L": ("--", 0.65)}

    for ax, name in zip(axes, CLASS_NAMES):
        color = CLASS_COLORS[LABEL_MAP[name]]
        for band, (ls, alpha) in styles.items():
            key = (band, name)
            if key not in psd_store:
                continue
            arr  = np.array(psd_store[key])           # (N, 1024)
            mean = arr.mean(axis=0)
            norm = np.linspace(0, 1, len(mean))       # normalised freq axis
            ax.plot(norm, mean, ls=ls, color=color, alpha=alpha,
                    lw=1.5, label=f"Band {band}")
            std = arr.std(axis=0)
            ax.fill_between(norm, mean - std, mean + std,
                            alpha=0.08, color=color)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Normalised frequency  (0 = DC, 1 = fs/2)")
        ax.set_ylabel("PSD (dB/Hz)")
        ax.legend(fontsize=8, framealpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "H-band vs L-band Mean PSD per Class\n"
        "(normalised freq axis — H: 40 MHz span, L: 10 MHz span)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir / "05_band_comparison.png")


# ---------------------------------------------------------------------------
# 7.  Train + evaluate
# ---------------------------------------------------------------------------

def train_xgboost(
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    out_dir: Path,
) -> dict:
    """
    Fit XGBoost on standardised scalar features.

    StandardScaler is fitted on train only — no leakage into val/test.
    val set is passed to XGBoost's eval_set for early stopping.
    """
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_va = scaler.transform(X[val_idx])
    X_te = scaler.transform(X[test_idx])

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    t0 = time.perf_counter()
    model.fit(
        X_tr, y[train_idx],
        eval_set=[(X_va, y[val_idx])],
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    best_iter = model.best_iteration
    print(f"  XGBoost: {elapsed:.1f}s  |  best iteration: {best_iter}")

    # ── Evaluation ──────────────────────────────────────────────────────────
    y_pred = model.predict(X_te)
    print(f"\n── XGBoost Classification Report ─────────────")
    print(classification_report(y[test_idx], y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm   = confusion_matrix(y[test_idx], y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — XGBoost (scalar features)")
    fig.tight_layout()
    _save(fig, out_dir / "06_confusion_matrix_xgb.png")

    # Feature importance
    importances = model.feature_importances_
    order       = np.argsort(importances)
    fig, ax     = plt.subplots(figsize=(7, 4))
    ax.barh(
        [SCALAR_NAMES[i] for i in order],
        importances[order],
        color=CLASS_COLORS[1], edgecolor="white",
    )
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("Feature Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, out_dir / "07_xgb_feature_importance.png")

    report_dict = classification_report(
        y[test_idx], y_pred,
        target_names=CLASS_NAMES, output_dict=True,
    )
    return {
        "model":       "XGBoost (scalars)",
        "accuracy":    round(report_dict["accuracy"], 4),
        "macro_f1":    round(report_dict["macro avg"]["f1-score"], 4),
        "best_iter":   best_iter,
        "train_time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# 8.  Main
# ---------------------------------------------------------------------------

def run(band: str = "H", max_segs: int|None = None, no_plots: bool = False) -> None:

    # ── Load ────────────────────────────────────────────────────────────────
    data      = load_dataset(H5_FILE, SPLIT_FILE, band=band, max_segs=max_segs)
    X_scalar  = data["X_scalar"]   # (N, 7)
    X_psd     = data["X_psd"]      # (N, 1024)
    y         = data["y"]
    freqs     = data["freqs"]
    meta      = data["meta"]
    splits    = data["splits"]

    train_idx = split_indices(meta, splits, "train")
    val_idx   = split_indices(meta, splits, "val")
    test_idx  = split_indices(meta, splits, "test")
    print(f"  Split — train: {len(train_idx)}, "
          f"val: {len(val_idx)}, test: {len(test_idx)}")

    # ── EDA ─────────────────────────────────────────────────────────────────
    if not no_plots:
        print(f"\n── EDA Plots → {FIG_DIR} ──────────────────────")
        plot_class_balance(meta, FIG_DIR)
        plot_mean_psd_per_class(X_psd, y, freqs, band, FIG_DIR)
        plot_example_spectrogram_per_class(H5_FILE, meta, band, FIG_DIR)
        plot_feature_boxplots(X_scalar, y, FIG_DIR)
        plot_band_comparison(H5_FILE, splits, FIG_DIR)

    # ── Train ───────────────────────────────────────────────────────────────
    print(f"\n── XGBoost ─────────────────────────────────────")
    result = train_xgboost(X_scalar, y, train_idx, val_idx, test_idx, FIG_DIR)
    result["band"] = band

    # ── Save results ────────────────────────────────────────────────────────
    pd.DataFrame([result]).to_csv(RES_FILE, index=False)
    print(f"\n── Results ──────────────────────────────────────")
    print(pd.DataFrame([result]).to_string(index=False))
    print(f"\n  Saved → {RES_FILE}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DroneRF EDA + DSP baseline pipeline")
    parser.add_argument("--band",     default="H", choices=["H", "L"],
                        help="Band to process (default: H, 40 MHz)")
    parser.add_argument("--max-segs", type=int, default=10,
                        help="Load at most N segments (smoke-test mode)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip EDA plots, only run the classifier")
    args = parser.parse_args()

    run(band=args.band, max_segs=args.max_segs, no_plots=args.no_plots)