import os
import argparse
import json
import time
from tqdm import tqdm
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless — safe on servers without a display
from scipy import signal as scipy_signal

import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Config / path helpers  (called inside main(), not at import time)
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file until we find the marker."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise RuntimeError(
        f"Could not find project root (looking for {marker}). "
        f"Started search from {current}"
    )


def _resolve_paths() -> tuple[Path, Path, Path]:
    root_dir   = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    h5_file    = Path(os.getenv("DATA_OUT_FILE")   or root_dir / "data/interim/dronerf.h5")
    split_file = Path(os.getenv("SPLIT_FILE") or root_dir / "data/interim/dronerf_splits.json")
    save_dir   = Path(os.getenv("FEATURE_OUT_FILE")   or root_dir / "data/processed/")

    return h5_file, split_file, save_dir


def _dsp_constants(cfg: dict) -> dict:
    """Unpack DSP config into a plain dict passed explicitly to DSP functions."""
    return dict(
        fs_h      = cfg["featurize"]["band_fs"]["H"],
        fs_l      = cfg["featurize"]["band_fs"]["L"],
        nperseg   = cfg["featurize"]["welch"]["nperseg"],
        noverlap  = cfg["featurize"]["welch"]["noverlap"],
        nfft      = cfg["featurize"]["welch"]["nfft"],
        window    = cfg["featurize"]["welch"]["window"],
        stft_nperseg  = cfg["featurize"]["stft"]["nperseg"],
        stft_noverlap = cfg["featurize"]["stft"]["noverlap"],
    )


# ---------------------------------------------------------------------------
# Constants that are derived formulas, not config decisions — module-level OK
# ---------------------------------------------------------------------------

SCALAR_NAMES = [
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
    "spectral_flatness",  "spectral_entropy",   "peak_freq", "snr_db",
]


# ---------------------------------------------------------------------------
# 1.  IQ helpers  (pure, no config dependency)
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
    nperseg: int,
    noverlap: int,
    nfft: int,
    window: str,
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
        window=window,
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
    nperseg: int,
    noverlap: int,
    nfft: int,
    window: str,
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
        window=window,
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
# 4.  Scalar features  (XGBoost input — pure, no config dependency)
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

    features: dict[str, float] = {
        "spectral_centroid":    centroid,
        "spectral_bandwidth":   bandwidth,
        "spectral_rolloff_85":  rolloff,
        "spectral_flatness":    flatness,
        "spectral_entropy":     entropy,
        "peak_freq":            peak_freq,
        "snr_db":               snr_db,
    }

    assert list(features.keys()) == SCALAR_NAMES, (
        f"SCALAR_NAMES mismatch.\n"
        f"  dict keys   : {list(features.keys())}\n"
        f"  SCALAR_NAMES: {SCALAR_NAMES}"
    )

    return np.array([features[k] for k in SCALAR_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# 5.  Main loader
# ---------------------------------------------------------------------------

def load_dataset(
    h5_path:    Path,
    split_path: Path,
    dsp:        dict,
    band:       list      = ["H", "L"],
    max_segs:   int|None  = None,
    save_dir:   Path|None = None,
    save_psd:   bool      = False,
) -> dict:
    """
    Load all segments for a given band, run DSP, return arrays.

    Parameters
    ----------
    dsp : dict
        DSP constants from _dsp_constants(cfg). Passed explicitly so this
        function carries no hidden dependency on global config state.

    Returns dict:
      X_scalar : (N, 7)    float32  — XGBoost features
      X_psd    : (N, 1024) float32  — full PSD (EDA only, None if save_psd=False)
      y        : (N,)      int32    — class labels
      freqs    : (1024,)   float32  — shared frequency axis (Hz)
      meta     : list[dict]         — segment provenance
      splits   : dict               — {"train": [...], "val": [...], "test": [...]}

    Saved layout (if save_dir is set):
      {save_dir}/
        {band}_scalar.npz   — X_scalar, y, freqs   (training input)
        {band}_psd.npz       — X_psd                (EDA only, if save_psd=True)
        {band}_meta.parquet  — meta + scalar columns (analysis)
    """
    band_fs = {"H": dsp["fs_h"], "L": dsp["fs_l"]}
    splits  = json.loads(split_path.read_text())
    all_ids = set(splits["train"] + splits["val"] + splits["test"])

    band_tag = "".join(sorted(band))

    X_scalar_list, X_psd_list, y_list, meta_list = [], [], [], []
    freqs_ref = None
    n_loaded  = 0

    t0 = time.perf_counter()

    with h5py.File(h5_path, "r") as hf:
        
        assert hf.attrs.get("signal_format") == "interleaved_iq", \
            "Unexpected signal_format — wrong H5 file?"
        assert "segments" in hf, "H5 missing 'segments' group"
        
        for key in tqdm(sorted(hf["segments"].keys())):
            if key not in all_ids:
                continue
            seg = hf[f"segments/{key}"]
            if seg.attrs["band"] not in band:
                continue

            raw           = seg["signal"][:]
            iq            = deinterleave(raw)
            fs            = band_fs[seg.attrs["band"]]
            freqs, psd_db = welch_psd(
                iq, fs=fs,
                nperseg=dsp["nperseg"],
                noverlap=dsp["noverlap"],
                nfft=dsp["nfft"],
                window=dsp["window"],
            )
            scalars = spectral_scalars(freqs, psd_db)

            if freqs_ref is None:
                freqs_ref = freqs

            X_scalar_list.append(scalars)
            if save_psd:
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

    X_scalar = np.array(X_scalar_list, dtype=np.float32)
    X_psd    = np.array(X_psd_list,    dtype=np.float32) if save_psd else None
    y        = np.array(y_list,         dtype=np.int32)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        scalars_path = save_dir / f"{band_tag}_scalar.npz"
        np.savez_compressed(scalars_path, X_scalar=X_scalar, y=y, freqs=freqs_ref)
        print(f"  → scalars : {scalars_path}")

        if save_psd and X_psd is not None:
            psd_path = save_dir / f"{band_tag}_psd.npz"
            np.savez_compressed(psd_path, X_psd=X_psd, y=y, freqs=freqs_ref)
            print(f"  → psd     : {psd_path}")

        meta_df   = pd.DataFrame(meta_list)
        scalar_df = pd.DataFrame(X_scalar, columns=SCALAR_NAMES)
        meta_df   = pd.concat([meta_df, scalar_df], axis=1)
        meta_path = save_dir / f"{band_tag}_meta.parquet"
        meta_df.to_parquet(meta_path, index=False)
        print(f"  → meta    : {meta_path}")

    return dict(
        X_scalar = X_scalar,
        X_psd    = X_psd,
        y        = y,
        freqs    = freqs_ref,
        meta     = meta_list,
        splits   = splits,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── 1. Environment and config ──────────────────────────────────────────
    load_dotenv()

    root_dir = Path(os.environ.get("PROJECT_ROOT", _find_project_root()))
    cfg      = _load_config(root_dir / "config/params.yaml")

    # ── 2. Path and DSP resolution ─────────────────────────────────────────
    h5_file, split_file, save_dir = _resolve_paths()
    dsp = _dsp_constants(cfg)

    # ── 3. CLI args ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Featurize DroneRF segments")
    parser.add_argument("--band",     default="H", choices=["H", "L"],
                        help="Band to process (default: H, 40 MHz)")
    parser.add_argument("--max-segs", type=int, default=None,
                        help="Load at most N segments (smoke-test only, omit for full run)")
    args = parser.parse_args()

    # ── 4. Run ────────────────────────────────────────────────────────────
    load_dataset(
        h5_path    = h5_file,
        split_path = split_file,
        dsp        = dsp,
        band       = [args.band],
        max_segs   = args.max_segs,
        save_dir   = save_dir,
        save_psd   = True,
    )


if __name__ == "__main__":
    main()