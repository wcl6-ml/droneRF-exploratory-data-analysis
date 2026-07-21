
import h5py
import tqdm

h5 = h5py.File(r"../data/interim/dronerf.h5", "r")
print(h5.keys())
print(h5["segments/00001"].attrs["band"])
#%%
print(sorted(h5["segments"].keys()))

for key in tqdm(sorted(h5["segments"].keys())):
    if key not in all_ids:
        continue
    seg = h5[f"segments/{key}"]
    if seg.attrs["band"] not in band:
        continue

    raw           = seg["signal"][:]

#%%
import numpy as np
scalars = np.load(r"../data/processed/H_scalars.npz", allow_pickle=False)