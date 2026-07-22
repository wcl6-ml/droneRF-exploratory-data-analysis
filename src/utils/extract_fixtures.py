# scripts/create_test_fixtures.py
import json
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from inference.predictor import DroneRFPredictor

PARQUET_PATH = "data/processed/H_meta.parquet"
H5_PATH = "data/interim/dronerf.h5"
SPLITS_PATH = "data/interim/dronerf_splits.json"

# fixture outputs
FIXTURE_NPZ_PATH = "tests/fixtures/multiclass_samples.npz"
FIXTURE_JSON_PATH = "tests/fixtures/expected_outputs.json"

def export_multiclass_fixtures():
    print("🔍 Reading splits and parquet metadata...")
    
    # 1. Load splits and grab only Validation / Train IDs (Excluding Test!)
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    
    # Prefer validation set for integration fixtures
    allowed_ids = set(splits.get("train", []))
    print(f"  ✓ Allowed pool size (val/train): {len(allowed_ids)} recording IDs")
    df = pd.read_parquet(PARQUET_PATH)
    
    # 2. Filter Parquet to ONLY include allowed recording IDs
    filtered_df = df[df["segment_id"].astype(str).isin(allowed_ids)]

    # Find all unique classes expected across the dataset
    all_expected_classes = df["drone_type"].unique()
    
    # Grab 1 representative sample per class from the filtered set
    sampled_df = filtered_df.groupby("drone_type").first().reset_index()
    # 3. VERIFICATION CHECK: Did we get a sample for every class?
    sampled_classes = set(sampled_df["drone_type"].unique())
    missing_classes = set(all_expected_classes) - sampled_classes
    
    if missing_classes:
        raise ValueError(
            f"❌ Missing samples for classes: {missing_classes}! "
            f"Check if all classes exist within your validation/train splits."
        )
    else:
        print(f"  ✓ Successfully found non-test samples for all {len(sampled_classes)} classes!")

    # 2. Warm up baseline predictor
    print("\n⚡ Initializing baseline predictor to record expected outputs...")
    
    model_path = Path("models/model.onnx")
    scaler_path = Path("models/scaler.json")
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


    fixture_dict = {}
    expected_outputs = {}
    
    with h5py.File(H5_PATH, "r") as h5:
        # Load metadata arrays once for fast lookup
        rec_ids = np.array([r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in h5["metadata"]["recording_id"][:]])
        bands = np.array([b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in h5["metadata"]["band"][:]])
        
        for _, row in sampled_df.iterrows():
            target_rec_id = str(row["recording_id"])
            label = str(row["drone_type"])
            
            rec_mask = (rec_ids == target_rec_id)
            h_indices = np.where(rec_mask & np.isin(bands, ["H", "H1", "H2"]))[0]
            l_indices = np.where(rec_mask & np.isin(bands, ["L", "L1", "L2"]))[0]
            
            if len(h_indices) == 0 or len(l_indices) == 0:
                raise KeyError(f"Could not locate matching H and L bands in HDF5 for recording_id: {target_rec_id}")

            raw_h = h5["segments"][f"{h_indices[0]:05d}"]['signal'][:]
            raw_l = h5["segments"][f"{l_indices[0]:05d}"]['signal'][:]
            
            # Save raw signals to dictionary
            fixture_dict[f"{label}_raw_h"] = raw_h
            fixture_dict[f"{label}_raw_l"] = raw_l
            
            # 3. Generate Snapshot Expected Output
            pred_result = predictor.predict(raw_h, raw_l)
            expected_outputs[label] = {
                "recording_id": target_rec_id,
                "expected_prediction": pred_result
            }
            print(f"  ✓ Class '{label}' snapshot output recorded: {pred_result['predicted_class']} ({pred_result['confidence']:.2f})")

    # 4. Save fixture assets
    Path(FIXTURE_NPZ_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Save raw arrays (.npz)
    np.savez_compressed(FIXTURE_NPZ_PATH, **fixture_dict)
    
    # Save snapshot expected results (.json)
    with open(FIXTURE_JSON_PATH, "w") as f:
        json.dump(expected_outputs, f, indent=2)

    print(f"\n🎉 Saved raw signals fixture: {FIXTURE_NPZ_PATH}")
    print(f"🎉 Saved baseline snapshot expected results: {FIXTURE_JSON_PATH}")

if __name__ == "__main__":
    export_multiclass_fixtures()