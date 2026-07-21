import h5py
import numpy as np

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

h5_path = "../data/interim/dronerf.h5"
results = get_paired_iq_signals(h5_path, target_seg_key="05000")
#%%