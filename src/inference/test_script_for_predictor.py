import h5py
import numpy as np
import json
from pathlib import Path
from src.inference.predictor import DroneRFPredictor

def generate_mock_iq_signal(num_samples: int) -> np.ndarray:
    """
    Generates a mock interleaved IQ signal.
    [I0, Q0, I1, Q1, ...] -> flat float32 array.
    """
    # A simple sine wave + random noise to simulate real physical input
    t = np.linspace(0, 1, num_samples // 2)
    i_ch = np.sin(2 * np.pi * 5e6 * t) + np.random.normal(0, 0.2, num_samples // 2)
    q_ch = np.cos(2 * np.pi * 5e6 * t) + np.random.normal(0, 0.2, num_samples // 2)
    
    # Interleave them
    interleaved = np.empty(num_samples, dtype=np.float32)
    interleaved[0::2] = i_ch
    interleaved[1::2] = q_ch
    return interleaved

def get_paired_iq_signals(h5_path: str, target_seg_key: str = "00000"):
    """
    Given a segment key (e.g. '00000'), finds its corresponding band, 
    looks up its recording_id, and locates the sibling segment for the other band.
    """
    with h5py.File(h5_path, "r") as h5:
        # 1. Convert key to integer index for metadata arrays
        target_idx = int(target_seg_key)
        
        # 2. Extract metadata for this target segment
        # Note: Decoding byte strings if stored as h5py string objects
        rec_id = h5["metadata"]["recording_id"][target_idx]
        if isinstance(rec_id, bytes): 
            rec_id = rec_id.decode("utf-8")
            
        band = h5["metadata"]["band"][target_idx]
        if isinstance(band, bytes): 
            band = band.decode("utf-8")

        print(f"Target segment '{target_seg_key}' belongs to Recording ID: {rec_id} | Band: {band}")

        # 3. Find all indices that share this recording_id
        all_rec_ids = np.array([
            r.decode("utf-8") if isinstance(r, bytes) else r 
            for r in h5["metadata"]["recording_id"][:]
        ])
        matching_indices = np.where(all_rec_ids == rec_id)[0]

        # 4. Filter for the H and L segments within those matching indices
        all_bands = np.array([
            b.decode("utf-8") if isinstance(b, bytes) else b 
            for b in h5["metadata"]["band"][:]
        ])

        h_idx = None
        l_idx = None

        for idx in matching_indices:
            if all_bands[idx] in ["H"]:  # Adapt to your exact string naming
                h_idx = idx
            elif all_bands[idx] in ["L"]:
                l_idx = idx

        if h_idx is None or l_idx is None:
            raise ValueError(f"Could not find both H and L segments for recording_id: {rec_id}")

        # 5. Load the raw IQ arrays from the segments dataset
        h_key = f"{h_idx:05d}"
        l_key = f"{l_idx:05d}"
        
        raw_h = h5["segments"][h_key]['signal'][:]
        raw_l = h5["segments"][l_key]['signal'][:]

        return raw_h, raw_l, rec_id

def test_prediction_pipeline():
    print("🚀 Initializing DroneRF Predictor Testing Harness...")
    
    # 1. Setup paths (Adjust these to where your artifacts actually live)
    model_path = Path("models/model.onnx")
    scaler_path = Path("models/scaler.json")
    h5_path = Path("data/interim/dronerf.h5")  # Adjust if your HDF5 is elsewhere
    # Quick sanity check: Create a dummy scaler if you don't have one generated yet
    if not scaler_path.exists():
        print("❌ Error: No scaler.json found! Please use the correct scaler generated during training .")

        
    if not model_path.exists():
        print(f"❌ Error: Place your trained ONNX model at '{model_path}' before running.")
        return

    # 2. Instantiate and load the predictor
    labels = ["Background", "Bebop", "AR-Drone", "Phantom"]
    predictor = DroneRFPredictor(
        model_path=model_path, 
        scaler_path=scaler_path, 
        label_names=labels
    )
    
    try:
        predictor.load()
        print("✅ Predictor loaded, ONNX session warm-booted successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return

    # 3. Simulate raw inputs for a single 1-second window
    # H-band: 40 MHz sample rate * 2 (interleaved I/Q) = 80,000,000 floats/sec
    # L-band: 10 MHz sample rate * 2 (interleaved I/Q) = 20,000,000 floats/sec
    print("\n📦 Generating simulated raw I/Q signal windows...")
    # raw_h = generate_mock_iq_signal(80_000_000)
    # raw_l = generate_mock_iq_signal(20_000_000)
    raw_h, raw_l, rec_id = get_paired_iq_signals(h5_path, target_seg_key="05000")
    print(f"   -> Raw H-Band buffer shape: {raw_h.shape} ({raw_h.dtype})")
    print(f"   -> Raw L-Band buffer shape: {raw_l.shape} ({raw_l.dtype})")
    print(f"   -> Associated Recording ID: {rec_id}")

    # 4. Run through the pipeline
    print("\n⚡ Running end-to-end inference pass (DSP -> Scaler -> ONNX)...")
    try:
        output = predictor.predict(raw_h, raw_l)
        print("\n🎉 Prediction Successful!")
        print("---------------------------------------")
        print(json.dumps(output, indent=2))
        print("---------------------------------------")
    except Exception as e:
        print(f"❌ Execution failed during preprocessing/inference: {e}")

if __name__ == "__main__":
    test_prediction_pipeline()