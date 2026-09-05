from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from kafka import KafkaConsumer

from detect.pipeline import run_detection_pipeline
from events.kafka_client import publish_fraud_event
from simulate.build_buckets import build_labeled_buckets

app = FastAPI()


# ============================================================
# YOUR EXISTING API ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# STAGE 2 MODEL REGISTRY
# ============================================================

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

METRICS_PATH = MODELS_DIR / "model_metrics.json"
FRAUD_TYPE_METRICS_PATH = MODELS_DIR / "fraud_type_metrics.json"

_model_cache: dict = {}

_active_model_id: str | None = None

_model_metrics: dict = {
    "default": None,
    "models": {}
}


def _load_metrics():

    global _model_metrics, _active_model_id

    if METRICS_PATH.exists():

        with open(METRICS_PATH) as f:
            _model_metrics = json.load(f)

        _active_model_id = _model_metrics.get("default")

    else:

        _model_metrics = {
            "default": None,
            "models": {}
        }

        _active_model_id = None


def _get_model(model_id: str):

    if model_id not in _model_cache:

        path = MODELS_DIR / f"stage2_{model_id}.pkl"

        _model_cache[model_id] = joblib.load(path)

    return _model_cache[model_id]


@app.get("/api/models")
def list_models():

    return {
        "active": _active_model_id,
        "models": _model_metrics.get("models", {}),
    }


@app.get("/api/models/{model_id}/evaluation")
def model_evaluation(model_id: str):

    model = _model_metrics.get("models", {}).get(model_id)
    if model is None:
        return {"error": f"Unknown model '{model_id}'"}

    evaluation = model.get("evaluation")
    if evaluation is None:
        return {
            "error": "Evaluation artifacts are missing. Run train_all_models.py and reload the dashboard."
        }

    return {
        "model_id": model_id,
        "label": model.get("label", model_id),
        "fraud_catch": model.get("fraud_catch"),
        **evaluation,
    }


@app.get("/api/fraud-type/evaluation")
def fraud_type_evaluation():

    if not FRAUD_TYPE_METRICS_PATH.exists():
        return {
            "error": "Fraud-type evaluation is missing. Run check/check_fraud_type_accuracy.py and reload the dashboard."
        }

    with open(FRAUD_TYPE_METRICS_PATH) as f:
        return json.load(f)


@app.post("/api/models/{model_id}/select")
def select_model(model_id: str):

    global _active_model_id

    if model_id not in _model_metrics.get("models", {}):
        return {"error": f"Unknown model '{model_id}'"}

    _get_model(model_id)  # warm the cache, fail fast if the .pkl is missing

    _active_model_id = model_id

    return {"active": _active_model_id}


# ============================================================
# PIPELINE RUN — uses whichever model is currently active
# ============================================================

@app.post("/api/pipeline/run")
def run_pipeline():

    if _active_model_id is None:
        return {"error": "No model selected. Run train_all_models.py first."}

    model = _get_model(_active_model_id)

    bucket = build_labeled_buckets()

    result = run_detection_pipeline(bucket, model=model)

    fraud_rows = result[result["stage2_pred"] == "fraud_attack"]

    for _, row in fraud_rows.iterrows():
        publish_fraud_event(row)

    by_type = (
        fraud_rows["fraud_type_pred"].value_counts().to_dict()
        if not fraud_rows.empty else {}
    )

    return {
        "model_used": _active_model_id,
        "total_buckets": int(len(result)),
        "stage1_flagged": int(result["anomaly_flag"].sum()),
        "stage2_fraud": int((result["stage2_pred"] == "fraud_attack").sum()),
        "stage2_demand": int((result["stage2_pred"] == "genuine_demand").sum()),
        "stage2_normal": int((result["stage2_pred"] == "normal").sum()),
        "published": int(len(fraud_rows)),
        "by_type": by_type,
    }


# ============================================================
# KAFKA → DASHBOARD BRIDGE
# ============================================================

FRAUD_TYPES = ["A", "B", "C", "D", "E"]

STAGES = [
    "problem",
    "acknowledgement",
    "action"
]

TOPICS = [
    f"fraud.type{t}.{stage}"
    for t in FRAUD_TYPES
    for stage in STAGES
]

_event_queue = queue.Queue()

_connected_clients: List[WebSocket] = []


def _kafka_listener():

    consumer = KafkaConsumer(
        *TOPICS,

        bootstrap_servers="localhost:9092",

        value_deserializer=lambda v:
            json.loads(v.decode("utf-8")),

        auto_offset_reset="latest",

        group_id="dashboard-bridge",
    )

    for message in consumer:

        _event_queue.put({
            "topic": message.topic,
            "data": message.value
        })


async def _broadcast_loop():

    loop = asyncio.get_event_loop()

    while True:

        try:
            payload = await loop.run_in_executor(
                None,
                _event_queue.get,
                True,
                1.0
            )

        except queue.Empty:
            continue

        dead = []

        for ws in _connected_clients:

            try:
                await ws.send_json(payload)

            except Exception:
                dead.append(ws)

        for ws in dead:

            if ws in _connected_clients:
                _connected_clients.remove(ws)


@app.on_event("startup")
async def _on_startup():

    _load_metrics()

    thread = threading.Thread(
        target=_kafka_listener,
        daemon=True
    )

    thread.start()

    asyncio.create_task(
        _broadcast_loop()
    )


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):

    await websocket.accept()

    _connected_clients.append(websocket)

    try:

        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:

        if websocket in _connected_clients:
            _connected_clients.remove(websocket)


# ============================================================
# DASHBOARD FRONTEND
# ============================================================

app.mount(
    "/",
    StaticFiles(
        directory="dashboard",
        html=True
    ),
    name="dashboard"
)
