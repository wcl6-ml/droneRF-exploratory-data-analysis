# DroneRF Edge API - Real-Time RF Drone Detection for Resource-Constrained Deployment

An edge-deployable RF drone classifier, served behind a REST API and packaged as a reproducible, cross-platform container. This built to demonstrate MLOps/deployment engineering, not model accuracy. The dataset is from [here](https://github.com/Al-Sad/DroneRF).

**Project Focus:** This project prioritizes **production-serving engineering**: a DVC-tracked reproducible pipeline, a serving architecture with a hard boundary between edge-deployable and other code, and cross-platform (amd64/arm64) container builds..

No physical SDR hardware is available for this project; a simulated sensor client replays real recordings from the dataset, window by window, the way a real edge device would consume one.

## Architecture Status

- DVC-tracked, reproducible pipeline (raw capture → features → trained model → ONNX export → deployment artifacts)
- Layered FastAPI service (`schemas.py` / `service.py` / `main.py`) — HTTP contract, model logic, and routing kept independently testable
- Hard separation between edge-deployable and other code — the deployed image has **zero** dependency on the raw dataset, `h5py`, or the simulator
- Single-window (non-batched) inference, matching how edge RF systems  consume a stream
- Cross-platform container build (native amd64, arm64 via QEMU emulation)
- Simulated sensor client for realistic replay, load generation, and integration testing
- To-do: Benchmarking harness run against real target hardware (e.g. Raspberry Pi), pending hardware access


## Data Format: `dronerf.h5`

`data_aggregator.py` converts the raw per-recording CSVs into a single self-contained
HDF5 archive plus a companion flat parquet for fast metadata queries.

**Hierarchy**
```

/segments/00000/signal float32 array, gzip-compressed  
/segments/00000/attrs (attrs: bui, drone_type, label, band, fs_hz,  
file_idx, seg_within_file, n_samples,  
signal_format, recording_id)  
/segments/00001/ ...  

/metadata/<column> flat per-column arrays, one row per segment  
(mirrors the record fields above)

```

Each segment is a fixed-length, non-overlapping slice of one raw recording. Segments are stored as individual HDF5 groups rather than one big flat array so that per-segment
provenance (band, sampling rate, source file) travels with the signal itself
and the archive stays self-describing without a separate lookup table.

**`recording_id`** (`{bui}_{file_idx}_{seg_within_file}`) is the key field:
it links the H-band and L-band segments that came from the *same physical
capture window*. It exists for two reasons:
- **Band fusion** — join H/L segments on `recording_id` for experiments that use both bands together.
- **Leak-free splitting** — segments from the same recording are temporally adjacent and highly correlated, so splitting by segment instead of by recording would let near-duplicate windows leak across train/val/test and inflate the reported metrics.

`fs_hz` is stored per segment (not just globally) because H-band and L-band
captures use the same sampling rates (40 MHz), just keeps this in case.

The companion `dronerf.meta.parquet` duplicates the `metadata/*` columns as a
flat table. It exists purely for speed: querying/grouping across ~thousands
of small HDF5 groups with `pandas` is slow, so `build_splits()` and any EDA
work read the parquet instead of touching the `.h5` file at all.

### Splitting Strategy

Splits are generated once by `build_splits()` and written to
`dronerf_splits.json` (segment IDs per split); every downstream stage reads
from this file instead of re-splitting, so train/val/test membership is
fixed across the whole pipeline.

---

## Key Project Structure

```
├── config
│   └── params.yaml
├── data
│   ├── interim
│   │   ├── dronerf.h5
│   │   ├── dronerf.meta.parquet
│   │   └── dronerf_splits.json
│   ├── processed
│   │   ├── H_meta.parquet
│   │   ├── H_scalar.npz
│   │   ├── H_scalars.npz
│   │   ├── L_meta.parquet
│   │   ├── L_psd.npz
│   │   ├── L_scalar.npz
│   │   └── L_scalars.npz
│   └── raw
├── models
│   ├── labels.json
│   ├── model.onnx
│   ├── model.onnx.data
│   ├── model.pt
│   └── scaler.json
├── packages
│   ├── drone-dev
│   │   ├── README.md
│   │   ├── dronerf-dev.Dockerfile
│   │   ├── pyproject.toml
│   │   ├── src
│   │   │   └── drone_dev
│   │   │       ├── __init__.py
│   │   │       ├── dump_sample_payload.py
│   │   │       ├── inference_time.py
│   │   │       └── simulate_sensor.py
│   │   └── uv.lock
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── data_aggregator.py
│   │   ├── featurize.py
│   │   └── unzip_raw.sh
│   ├── models
│   │   └── model.py
│   ├── train
│   │   ├── eval_onnx.py
│   │   └── train.py
└── tests
    └── fixtures
        ├── expected_outputs.json
        ├── multiclass_samples.npz
        └── test_script_for_predictor.py

```
* `config/params.yaml` - single source of truth for parameters at different stage.
* `src/utils/generate_labels.py` -  one-time: `params.yaml` -> `models/labels.json` for edge device (DVC staged).
* `models/` -  `model.onnx, model.onnx.data, scaler.json, labels.json`.
* `dvc.yaml / dvc.lock` - full reproducible pipeline.

## Getting Started
### 1. Place the dataset
Move the dataset .zip file to `./data/raw/`. Unzip the dataset:
```
unzip DATASET.zip
```
Will give the folder tree:
```
data/raw/
└── DroneRF
    ├── AR drone
	    ...
	├── Background RF activites
		...
	├── Bepop drone
		...
	└── Phantom drone
		...
```
### 2. Unzip the .rar files within each category
```
./src/data/unzip_raw.sh
```

### 3. Reproduce the pipeline
The pipeline from data aggregation, feature engineering, traini and evaluation of the model are llisted in `dvc.yaml`. 

Run the `dvc repro` to reproduce the results. 
```bash
dvc repro
```

This runs the full pipeline end to end, data aggregation, feature/scalar extraction, training, ONNX export, and `labels.json` generation,  and produces everything under `models/` (`model.onnx`, `model.onnx.data`, `scaler.json`, `labels.json`) that the edge container needs.

### 2. Build and run the edge container

```bash
docker build -t drone-edge:latest -f packages/drone-edge/Dockerfile .

docker run -p 8000:22111 drone-edge:latest

curl http://0.0.0.0:8000/health
```

Cross-platform build (native amd64 + emulated arm64 via Buildx/QEMU):

```bash
docker buildx build --platform linux/arm64 -t drone-edge/arm64:latest -f packages/drone-edge/Dockerfile .
```

### 3. Exercise it with the simulated sensor client

```bash
python -m packages.drone-devd.rone_dev.simulate_sensor \
  --scenario "background:20,bebop:10,ar:10,background:20,phantom:10" \
  --speed 50 \
  --api-url http://0.0.0.0:8000
```

This replays real, physically continuous recordings against the running API in a chosen class sequence and reports predicted vs. ground-truth label per window, the same shape of integration test a real edge deployment would run.

---
## API Usage

**Endpoints:** `GET /health`, `GET /v1/model/info`, `POST /v1/predict`

**Note**: What `/predic` takes
The pipeline:
1. Raw RF signals
2. Processed scalars
3. L/H bands concatenated

Generate one example of paylod for testing
```
python packages/drone-dev/src/drone_dev/dump_sample_payload.py
```

```bash
curl -X POST http://0.0.0.0:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d @payload.json
```

**Example response:**

```json
{
"predicted_class":"ar",
"class_index":2,
"confidence":0.5291,
"inference_time_ms":15.383,
"model_version":"v1.0.0"
}
```

---
## Benchmarking Tooling

`scripts/bench.sh` measures image size, cold start to `/health`, idle/peak RSS under simulated load, and server-side inference latency. 

```
./scripts/bench.sh
```

| Metric                                 | Native x86_64 |
| -------------------------------------- | ------------- |
| Image size (`docker save`)             | 145M          |
| Image size (`docker images`)           | 611MB         |
| Cold start (to `/health`)              | 11.21s        |
| RSS, idle                              | 351.7MiB      |
| RSS, peak under load                   | 350.2MB       |
| Inference latency, p50 (server-side)   | 17.56 ms      |
| Inference latency, p95 (server-side)   | 24.32 ms      |
| Inference latency, p99 (server-side)   | 29.36 ms      |
| End-to-end client latency, p50 (`hey`) | 41.4 ms       |
| End-to-end client latency, p95 (`hey`) | 52.1 ms       |
 An end-to-end client latency (via `hey`) against a running container.
```
docker run -p 8000:22111 drone-edge:latest

hey -n 200 -c 1 -m POST -T "application/json" -D payload.json http://0.0.0.0:8000/v1/predict
```

**arm64:** the image builds correctly via `docker buildx build --platform linux/arm64` and was functionally verified end-to-end (boots, `/health` reports healthy, `/v1/predict` returns correct predictions) by running it under QEMU emulation on this same x86_64 host. Those emulated numbers aren't reported here. QEMU adds significant CPU overhead on top of the architecture difference, so they wouldn't reflect real arm64 performance (e.g. Raspberry Pi). The pipeline is ready to produce real numbers as soon as target hardware is available; what's confirmed today is correctness, not speed, on arm64.

Given the model is a small 1D-CNN over 14 scalar features, note that these numbers mostly reflect the serving stack's overhead (Docker, FastAPI, ONNX Runtime startup) rather than model compute, the point is the harness and the discipline of measuring before shipping, not raw model speed.


---

# Reference
* Al-Sa'd, Mohammad; Allahham, Mhd Saria; Mohamed, Amr; Al-Ali, Abdulla; Khattab, Tamer; Erbad, Aiman (2019), “DroneRF dataset: A dataset of drones for RF-based detection, classification, and identification”, Mendeley Data, v1. [http://dx.doi.org/10.17632/f4c2b4n755.1](http://dx.doi.org/10.17632/f4c2b4n755.1)


