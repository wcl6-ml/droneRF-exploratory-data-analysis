"""
main.py
=======
The HTTP layer. Deliberately thin: parse request -> call service.py ->
serialize response. No DSP, no ONNX, no dataset knowledge lives here --
service.py is treated as a black box that "just does inference."

Endpoints
---------
GET  /health           liveness/readiness check
GET  /v1/model/info     class list + model metadata
POST /v1/predict        classify one raw H/L window

Run:
    uvicorn drone_edge.main:app --reload --port 8000
Interactive docs (auto-generated from schemas.py) at:
    http://localhost:8000/docs
"""
import os
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from drone_edge.schemas import (
    PredictRequest,
    PredictResponse,
    ModelInfoResponse,
    HealthResponse,
)
from drone_edge.service import ModelService

MODEL_PATH = "models/model.onnx"
SCALER_PATH = "models/scaler.json"
LABELS_PATH = "models/labels.json"
MODEL_VERSION = "v1.0.0"
SEGMENT_LENGTH = 100_000  # must match config/params.yaml's data_aggregator.segment_length

PORT = int(os.getenv("PREDICTOR_API", "22111"))
HOST = os.getenv("PREDICTOR_HOST", "0.0.0.0")


# Holds the one, process-wide ModelService instance. A plain dict (not a
# bare global variable) so it's explicit and easy to clear in tests.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs exactly once, when the server process starts -- not per
    request. This is where the ~seconds-long cost of loading the ONNX
    session happens, so no single request ever pays for it.
    """
    state["service"] = ModelService(MODEL_PATH, SCALER_PATH, LABELS_PATH).load()
    yield
    state.clear()


app = FastAPI(title="DroneRF Classifier API", version=MODEL_VERSION, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness probe. Answers 'is this instance up and able to serve?'"""
    ready = "service" in state and state["service"].is_ready
    return {"status": "ok" if ready else "not_ready", "model_loaded": ready}


@app.get("/v1/model/info", response_model=ModelInfoResponse)
def model_info():
    """
    Lets a client discover the class list and expected input size at
    runtime, instead of hardcoding a guess that can drift from the
    actual deployed model.
    """
    service: ModelService = state["service"]
    return {
        "classes": service.classes,
        "model_version": MODEL_VERSION,
        "segment_length": SEGMENT_LENGTH,
    }


@app.post("/v1/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Classifies one raw H/L window.

    Note the two distinct failure modes here, and why they're not
    handled the same way:
      - Missing/wrong-typed fields in the request body never reach this
        function at all -- Pydantic (via PredictRequest) already
        rejected them with a 422 before predict() was called. That's a
        *structural* error: "your request doesn't match the contract."
      - Anything that fails *inside* service.predict() (e.g. a window
        too short for the Welch PSD to run) is a *semantic* error: the
        request was well-formed, but processing it failed. That's what
        the try/except below turns into a 400, with the underlying
        reason in the response so the caller can actually debug it.
    """
    service: ModelService = state["service"]
    try:
        result = service.predict(req.raw_h, req.raw_l)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return {**result, "model_version": MODEL_VERSION}

def main():
    """Entry point for the application when run as a script"""
    uvicorn.run(
        "drone_edge.main:app",  # The import string path
        host=HOST, 
        port=PORT, 
        reload=True,
        #app_dir="src"           
    )

if __name__ == "__main__":
    main()

