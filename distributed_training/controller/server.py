from fastapi import FastAPI, Request
from typing import List, Dict, Union
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import random
import platform
import socket
from zeroconf import ServiceBrowser, Zeroconf

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev — lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# === Models ===
class WorkerStatus(BaseModel):
    id: str
    status: str
    progress: int
    accuracy: Union[float, None]
    loss: Union[float, None]
    latency: Union[float, None]
    power: Union[float, None]
    storage: Union[float, None]
    ram: Union[float, None]
    aurora: Union[float, None]

class NodeRegistration(BaseModel):
    id: str
    role: Union[str, None]  # "trainer", "inference", "dashboard"
    hardware: Dict[str, Union[str, float, bool]]
    tags: List[str] = []

class TaskRequest(BaseModel):
    type: str  # "train" or "infer"
    data: dict = {}

# === In-memory Registry ===
node_registry: Dict[str, Dict] = {}

# === Endpoints ===
@app.post("/register")
def register_node(payload: NodeRegistration):
    role = payload.role or classify_node(payload.hardware)
    tags = payload.tags or generate_tags(payload.hardware)
    node_registry[payload.id] = {
        "role": role,
        "hardware": payload.hardware,
        "tags": tags,
        "url": f"http://{payload.hardware['device']}:9000" if role != "dashboard" else None,
        "status": "idle",
        "progress": 0
    }
    return {"message": f"Node {payload.id} registered successfully as {role}."}

@app.get("/nodes")
def list_nodes():
    return node_registry

@app.post("/distribute")
def distribute_training():
    results = []
    for node_id, node in node_registry.items():
        if node['role'] != 'trainer':
            continue
        try:
            url = node['url'] + "/train"
            res = requests.post(url, json={"type": "train", "data": {}})
            results.append(res.json())
        except Exception as e:
            results.append({"id": node_id, "error": str(e)})
    return {"results": results}

@app.get("/status", response_model=List[WorkerStatus])
def get_status():
    status_list = []
    for node_id, node in node_registry.items():
        try:
            if node["url"]:
                res = requests.get(f"{node['url']}/status")
                status_list.append(res.json())
        except:
            status_list.append({
                "id": node_id,
                "status": "offline",
                "progress": 0,
                "accuracy": None,
                "loss": None,
                "latency": None,
                "power": None,
                "storage": None,
                "ram": None,
                "aurora": None
            })
    return status_list

class DeviceListener:
    def __init__(self):
        self.devices = []

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info:
            ip = socket.inet_ntoa(info.addresses[0])
            hostname = name.split('.')[0]
            self.devices.append({
                "ip": ip,
                "hostname": hostname,
                "ram": None,  # Optional: make a follow-up API call to get RAM info
                "gpu": None
            })
            
@app.get("/scan")
def scan_local_network():
    zeroconf = Zeroconf()
    listener = DeviceListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)

    import time
    time.sleep(3)  # Wait to discover services
    zeroconf.close()
    return {"devices": listener.devices}

@app.post("/start-training")
def start_training(config: dict):
    print("[Controller] Starting distributed training with config:", config)
    results = []

    for node_id, node in node_registry.items():
        if node['role'] == 'training':
            try:
                url = node['url'] + "/train"
                res = requests.post(url, json={"type": "train", "data": {"config": config}})
                results.append(res.json())
            except Exception as e:
                results.append({"id": node_id, "error": str(e)})

    return {"results": results}
