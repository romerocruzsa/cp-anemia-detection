from fastapi import FastAPI
from pydantic import BaseModel
import requests
import random
import platform
import time
import os
import socket

def get_device_specs():
    return {
        "ram": round(random.uniform(1024, 8192), 2),  # in MB
        "gpu": random.random() > 0.4,
        "cpu": platform.processor(),
        "device": platform.node(),
        "ip": socket.gethostbyname(socket.gethostname())
    }

def classify_node(specs: dict) -> str:
    """
    Classify device as 'trainer', 'inference', or 'dashboard'.
    """
    ram = specs.get("ram", 0)
    has_gpu = specs.get("gpu", False)

    if ram >= 4096 and has_gpu:
        return "trainer"
    elif ram >= 512:
        return "inference"
    else:
        return "dashboard"

def generate_tags(specs: dict) -> list:
    tags = []
    if specs.get("gpu"):
        tags.append("gpu")
    if specs.get("ram", 0) >= 4096:
        tags.append("high-mem")
    elif specs.get("ram", 0) < 1024:
        tags.append("low-resource")
    tags.append("auto")
    return tags

app = FastAPI()

# === Node Config ===
NODE_ID = os.getenv("NODE_ID", f"node-{random.randint(1000, 9999)}")
# ROLE = os.getenv("ROLE", "trainer")  # 'trainer', 'inference', 'dashboard'
CONTROLLER_URL = os.getenv("CONTROLLER_URL", "http://localhost:8000")

# === Hardware Specs (mocked for now) ===
HARDWARE = get_device_specs()
ROLE = classify_node(HARDWARE)

TRAINING_HISTORY = []

# === Data Models ===
class Task(BaseModel):
    type: str
    data: dict = {}

# === Node Registration ===
@app.on_event("startup")
def register_with_controller():
    payload = {
        "id": NODE_ID,
        "role": ROLE,
        "hardware": HARDWARE,
        "tags": generate_tags(HARDWARE)
    }
    try:
        res = requests.post(f"{CONTROLLER_URL}/register", json=payload)
        print(f"[✓] Registered as {NODE_ID} ({ROLE}) → {res.status_code}")
    except Exception as e:
        print(f"[x] Registration failed: {e}")

# === ML Task Endpoints ===
@app.post("/train")
def train_model(task: Task):
    print(f"[{NODE_ID}] Received training task")

    try:
        # from main import main as run_main_training
        # run_main_training()
        time.sleep(60*2)

        result = {
            "id": NODE_ID,
            "status": "complete",
            "accuracy": round(random.uniform(0.85, 0.9), 4),
            "loss": round(random.uniform(0.1, 0.2), 4),
            "latency": round(random.uniform(120, 180), 2),
            "power": round(random.uniform(2.5, 3.5), 2),
            "ram": HARDWARE["ram"],
            "storage": round(random.uniform(40, 60), 2),
            "history": [round(random.uniform(0.85, 0.9), 4) for _ in range(10)]
        }
        TRAINING_HISTORY.append(result)
        return result

    except Exception as e:
        print(f"[{NODE_ID}] Error during training: {e}")
        return {"error": str(e)}

@app.post("/infer")
def infer_model(task: Task):
    time.sleep(1)
    return {
        "id": NODE_ID,
        "status": "complete",
        "output": random.choice(["positive", "negative"]),
        "confidence": round(random.uniform(0.7, 0.99), 4)
    }

@app.get("/status")
def get_status():
    last = TRAINING_HISTORY[-1] if TRAINING_HISTORY else {}
    return {
        "id": NODE_ID,
        "status": "idle",
        "role": ROLE,
        "progress": random.randint(10, 100),
        "accuracy": last.get("accuracy"),
        "loss": last.get("loss"),
        "latency": last.get("latency"),
        "power": last.get("power"),
        "ram": HARDWARE["ram"],
        "storage": round(random.uniform(40, 60), 2),
        "aurora": last.get("aurora"),
        "history": [entry["accuracy"] for entry in TRAINING_HISTORY[-10:]]
    }
