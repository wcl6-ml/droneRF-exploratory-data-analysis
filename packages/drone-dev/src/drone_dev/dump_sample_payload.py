import json
from pathlib import Path
from simulate_sensor import replay

stream = replay(
    h5_path=Path("data/interim/dronerf.h5"),
    split_file=Path("data/interim/dronerf_splits.json"),
    split="test",
    mode="shuffled",
    speed_factor=1e9,  # don't actually wait between yields
)
window = next(stream)
payload = {"raw_h": window["raw_h"].tolist(), "raw_l": window["raw_l"].tolist()}
Path("payload.json").write_text(json.dumps(payload))
print(f"Dumped one window from {window['recording_id']} (truth={window['drone_type']})")

