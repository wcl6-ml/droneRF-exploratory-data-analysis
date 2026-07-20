import os
import json
import numpy as np
import onnxruntime as ort
from scipy import signal as scipy_signal
from pathlib import Path
from typing import Tuple, Dict, Any, Union

class DroneRFPredictor:
    """
    Standalone Edge Engine for DroneRF.
    Transforms raw I/Q signal data from H and L bands into scaled scalar feature 
    tensors, then executes ONNX Runtime inference.
    """
    def __init__(
        self, 
        model_path: Union[str, Path] = None, 
        scaler_path: Union[str, Path] = None,
        label_names: list[str] = None
    ):
        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/model.onnx"))
        self.scaler_path = Path(scaler_path or os.getenv("SCALER_PATH", "models/scaler.json"))
        self.label_names = label_names or ["background", "drone_type_a", "drone_type_b", "drone_type_c"] 
        
        # Hardcoded production DSP configurations derived from featurize.py params
        self.dsp_params = {
            "fs_h": 40e6,      # H-band: 40 MHz sample rate
            "fs_l": 10e6,      # L-band: 10 MHz sample rate
            "nperseg": 1024,
            "noverlap": 512,
            "nfft": 1024,
            "window": "hamming"
        }
        
        self.scalar_names = [
            "spectral_centroid", "spectral_bandwidth", "spectral_rolloff_85",
            "spectral_flatness", "spectral_entropy", "peak_freq", "snr_db"
        ]
        
        self.session = None
        self.input_name = None
        self.mean = None
        self.std = None

    def load(self) -> "DroneRFPredictor":
        """Loads scaling matrices and warm-boots ONNX session."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler missing at: {self.scaler_path.resolve()}")
        scaler_data = json.loads(self.scaler_path.read_text())
        self.mean = np.array(scaler_data["mean"], dtype=np.float32)
        self.std = np.array(scaler_data["std"], dtype=np.float32)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model missing at: {self.model_path.resolve()}")
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        return self

    # ── DSP CORE TRANSFORMS (Copied from featurize.py) ──────────────────────

    def _deinterleave(self, x: np.ndarray) -> np.ndarray:
        """[I0, Q0, I1, Q1...] -> complex64 array."""
        return (x[0::2] + 1j * x[1::2]).astype(np.complex64)

    def _welch_psd(self, iq: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates centered Welch PSD in dB."""
        freqs, psd = scipy_signal.welch(
            iq, fs=fs,
            window=self.dsp_params["window"],
            nperseg=self.dsp_params["nperseg"],
            noverlap=self.dsp_params["noverlap"],
            nfft=self.dsp_params["nfft"],
            return_onesided=False,
            scaling="density",
        )
        freqs = np.fft.fftshift(freqs)
        psd_db = 10 * np.log10(np.fft.fftshift(np.abs(psd)) + 1e-12)
        return freqs.astype(np.float32), psd_db.astype(np.float32)

    def _extract_spectral_scalars(self, freqs: np.ndarray, psd_db: np.ndarray) -> np.ndarray:
        """Extracts the 7 base scalar features from a single band's PSD profile."""
        psd_lin = 10 ** (psd_db / 10.0)
        psd_norm = psd_lin / (psd_lin.sum() + 1e-12)

        centroid = float(np.sum(freqs * psd_norm))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm)))
        cumsum = np.cumsum(psd_norm)
        rolloff = float(freqs[np.searchsorted(cumsum, 0.85)])
        flatness = float(np.exp(np.mean(np.log(psd_lin + 1e-12))) / (np.mean(psd_lin) + 1e-12))
        entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-12)))
        peak_idx = int(np.argmax(psd_lin))
        peak_freq = float(freqs[peak_idx])
        snr_db = float(psd_db[peak_idx] - np.median(psd_db))

        features = {
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff_85": rolloff,
            "spectral_flatness": flatness,
            "spectral_entropy": entropy,
            "peak_freq": peak_freq,
            "snr_db": snr_db,
        }
        return np.array([features[k] for k in self.scalar_names], dtype=np.float32)

    # ── END-TO-END PREPROCESSING ──────────────────────────────────────────

    def _preprocess(self, raw_h_signal: np.ndarray, raw_l_signal: np.ndarray) -> np.ndarray:
        """
        Takes raw, flat I/Q arrays from both bands, extracts features,
        concatenates them into a 14-element vector, and scales it.
        """
        # 1. Process H-Band
        iq_h = self._deinterleave(raw_h_signal)
        freqs_h, psd_h = self._welch_psd(iq_h, self.dsp_params["fs_h"])
        scalars_h = self._extract_spectral_scalars(freqs_h, psd_h)

        # 2. Process L-Band
        iq_l = self._deinterleave(raw_l_signal)
        freqs_l, psd_l = self._welch_psd(iq_l, self.dsp_params["fs_l"])
        scalars_l = self._extract_spectral_scalars(freqs_l, psd_l)

        # 3. Concatenate bands to build the flat 14-element row (H_scalars + L_scalars)
        X_flat = np.hstack([psd_h, psd_l]).astype(np.float32)
        X_len = X_flat.shape[-1]

        # 4. Apply Z-score transformation: (X - mean) / std
        X_scaled = (X_flat - self.mean) / self.std
        
        # 5. Reshape to 3D tensor format for 1D CNN: (1, 1, 14)
        return X_scaled.reshape(1, 1, X_len)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, raw_h_signal: np.ndarray, raw_l_signal: np.ndarray) -> Dict[str, Any]:
        if self.session is None:
            raise RuntimeError("Predictor session is not loaded. Call .load() first.")
            
        # Run end-to-end DSP + scaling pipeline
        input_tensor = self._preprocess(raw_h_signal, raw_l_signal)
        
        # Inference pass
        raw_logits = self.session.run(None, {self.input_name: input_tensor})[0]
        
        # Calculate scores
        probabilities = self._softmax(raw_logits)[0]
        predicted_idx = int(np.argmax(probabilities))
        
        return {
            "predicted_class": self.label_names[predicted_idx],
            "class_index": predicted_idx,
            "confidence": round(float(probabilities[predicted_idx]), 4)
        }