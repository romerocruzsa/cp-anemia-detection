import os
import sys

# Dynamically add root path (cp-anemia-detection) to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
from fastapi import FastAPI, Request
from typing import List, Dict, Union
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import random
import platform
import socket
import json
import time
from fastapi import APIRouter
import socket
from zeroconf import ServiceBrowser, Zeroconf, ServiceStateChange, ServiceListener

router = APIRouter()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev — lock this down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

discovered_devices = []

def on_service_state_change(zeroconf, service_type, name, state_change):
    if state_change is ServiceStateChange.Added:
        info = zeroconf.get_service_info(service_type, name)
        if info:
            hostname = name.split('.')[0]
            ip = ".".join(str(b) for b in info.addresses[0])
            discovered_devices.append({
                "id": hostname,
                "ip": ip,
                "ram": 2048,  # Placeholder — replace with actual later
                "gpu": False  # Placeholder — replace with actual later
            })

NODES_FILE = os.path.join(os.path.dirname(__file__), "../configs/nodes.json")

def update_nodes_json(role, node_info):
    with open(NODES_FILE, "r+") as f:
        data = json.load(f)

        if role == "controller":
            data["controller"] = node_info["url"]
        elif role == "receiver":
            data["receiver"] = node_info["url"]
        elif role in ("trainer", "inference"):
            if node_info["url"] and node_info["url"] not in data["workers"]:
                data["workers"].append(node_info["url"])

        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()

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
class FastScanListener(ServiceListener):
    def __init__(self):
        self.devices = []

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])
            hostname = name.split('.')[0]
            self.devices.append({
                "id": hostname,
                "ip": ip,
                "ram": 2048,
                "gpu": False
            })
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
# controller/server.py
@app.on_event("startup")
def self_register():
    try:
        payload = {
            "id": "controller",
            "role": "training",  # ✅ Mark it as a trainer
            "hardware": {
                "ram": 8192,
                "gpu": True,
                "cpu": platform.processor(),
                "device": "localhost"
            },
            "tags": ["controller", "trainer"]
        }

        node_registry["controller"] = {
            "role": payload["role"],
            "hardware": payload["hardware"],
            "tags": payload["tags"],
            "url": "http://localhost:8000",  # or whatever port controller uses
            "status": "idle",
            "progress": 0
        }

        update_nodes_json("trainer", {"url": "http://localhost:8000"})  # also update workers list
        print("[✓] Controller registered as trainer.")
    except Exception as e:
        print("[x] Controller registration failed:", e)

@app.post("/register")
def register_node(payload: NodeRegistration):
    if payload.id in node_registry:
            print(f"[Controller] Node {payload.id} already registered.")
            return {"message": f"Node {payload.id} already registered."}

    role = payload.role or classify_node(payload.hardware)
    tags = payload.tags or generate_tags(payload.hardware)
    url = f"http://{payload.hardware['device']}:9000" if role != "dashboard" else None

    node_registry[payload.id] = {
        "role": role,
        "hardware": payload.hardware,
        "tags": tags,
        "url": url,
        "status": "idle",
        "progress": 0
    }

    update_nodes_json(role, {"url": url})
    return {"message": f"Node {payload.id} registered successfully as {role}."}

@app.get("/nodes")
def list_nodes():
    return node_registry

@app.post("/distribute")
def distribute_training():
    results = []
    for node_id, node in node_registry.items():
        if node['role'] != 'training':
            continue
        try:
            if node_id == "controller":
                from main import main
                main(config=None)  # 🧠 Optionally pass config here
                results.append({"id": "controller", "status": "trained locally"})
            else:
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

@router.get("/scan")
def scan_devices():
    try:
        # Always include the controller
        controller_device = {
            "id": "controller",
            "ip": "localhost",
            "ram": 8192,
            "gpu": True
        }

        zeroconf = Zeroconf()
        listener = FastScanListener()
        browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)

        time.sleep(2)  # Wait for discovery
        zeroconf.close()

        discovered = listener.devices
        print(f"[Scan] Found {len(discovered)} devices via Zeroconf.")
        return {"devices": [controller_device] + discovered}

    except Exception as e:
        print(f"[Scan] Zeroconf error: {e}")
        return {"devices": [{
            "id": "controller",
            "ip": "localhost",
            "ram": 8192,
            "gpu": True
        }]}

# @app.get("/scan")
# def scan_devices():
#     controller_device = {
#         "id": "controller",
#         "ip": "localhost",
#         "ram": 8192,
#         "gpu": True
#     }

#     other_devices = [
#         {"id": "nvj1", "ip": "192.168.1.101", "ram": 4096, "gpu": True},
#         {"id": "pi-edge1", "ip": "192.168.1.102", "ram": 1024, "gpu": False},
#         {"id": "ios-13", "ip": "192.168.1.103", "ram": 2048, "gpu": False}
#     ]

#     return {"devices": [controller_device] + other_devices}

@app.post("/start-training")
def start_training(config: dict):
    print("[Controller] Starting distributed training with config:", config)
    results = []

    for node_id, node in node_registry.items():
        if node['role'] == 'training':
            if node_id == "controller":
                print("[Controller] Running local training...")
                try:
                    from jobs.manager import main
                    output = main(config)
                    results.append({"id": node_id, "result": output})
                except Exception as e:
                    results.append({"id": node_id, "error": str(e)})
            else:
                try:
                    url = node['url'] + "/train"
                    res = requests.post(url, json={"type": "train", "data": {"config": config}})
                    results.append(res.json())
                except Exception as e:
                    results.append({"id": node_id, "error": str(e)})

    return {"results": results}

app.include_router(router)
