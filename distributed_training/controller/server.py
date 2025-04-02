from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev — lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WorkerStatus(BaseModel):
    id: str
    status: str
    progress: int
    accuracy: float | None
    loss: float | None
    latency: float | None
    power: float | None
    storage: float | None
    ram: float | None
    aurora: float | None

WORKER_PORTS = [9000, 9001]

@app.post("/distribute")
def distribute_training():
    results = []
    for port in WORKER_PORTS:
        try:
            res = requests.post(f"http://localhost:{port}/train")
            results.append(res.json())
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}

@app.get("/status", response_model=List[WorkerStatus])
def get_status():
    return [
        {"id": "nvj0", "status": "active", "progress": 80, "accuracy": 0.87, "loss": round(random.uniform(0.1, 0.2), 4), "latency":round(random.uniform(120, 180), 2),
                                                                            "power": round(random.uniform(2.5, 3.5), 2), "ram":round(random.uniform(128, 256), 2),
                                                                            "storage":round(random.uniform(40, 60), 2), "aurora": round(random.uniform(2.25, 3.45), 4)},
        {"id": "nvj1", "status": "active", "progress": 45, "accuracy": 0.83, "loss": round(random.uniform(0.1, 0.2), 4), "latency":round(random.uniform(120, 180), 2),
                                                                            "power": round(random.uniform(2.5, 3.5), 2), "ram":round(random.uniform(128, 256), 2),
                                                                            "storage":round(random.uniform(40, 60), 2), "aurora": round(random.uniform(2.25, 3.45), 4)},
        {"id": "pi-edge1", "status": "idle", "progress": 0, "accuracy": None, "loss": None, "latency": None, "power": None, "ram": None, "storage": None, "aurora": None},
        {"id": "ios-13", "status": "offline", "progress": 0, "accuracy": None, "loss": None, "latency": None, "power": None, "ram": None, "storage": None, "aurora": None},
    ]
       