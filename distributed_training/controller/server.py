from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

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
        {"id": "nvj0", "status": "active", "progress": 80, "accuracy": 0.87, "loss": 0.13},
        {"id": "nvj1", "status": "active", "progress": 45, "accuracy": 0.83, "loss": 0.17},
        {"id": "pi-edge1", "status": "idle", "progress": 0, "accuracy": None, "loss": None},
        {"id": "ios-13", "status": "offline", "progress": 0, "accuracy": None, "loss": None}
    ]