import numpy as np
from pathlib import Path
from simulate_sensor import replay, check_api_health, send_window

API = "http://localhost:8787"
check_api_health(API)

stream = replay(
    h5_path=Path("data/interim/dronerf.h5"),
    split_file=Path("data/interim/dronerf_splits.json"),
    split="test",
    mode="shuffled",
    speed_factor=1e9,  # don't actually wait between yields
)
times = [send_window(w, API)["inference_time_ms"] for _, w in zip(range(200), stream)]

print(f"inference p50={np.percentile(times, 50):.2f}ms  p95={np.percentile(times, 95):.2f}ms")
