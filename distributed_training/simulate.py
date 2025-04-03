import os
import subprocess
import time
import requests
import json
import webbrowser

def launch_uvicorn(app_path, port, cwd):
    return subprocess.Popen(
        ['uvicorn', app_path, '--port', str(port), '--reload'],
        cwd=cwd
    )

def load_config():    
    config_path = os.path.join(os.path.dirname(__file__), "configs/nodes.json")
    with open(config_path) as f:
        return json.load(f)

# Load configuration
config = load_config()
controller_url = config["controller"]
worker_nodes = config["workers"]
receiver_url = config["receiver"]

# # Step 0 — Launch result receiver
# print("[0] Launching result receiver...")
# receiver_proc = launch_uvicorn("receiver:app", 8500, "receiver")
# time.sleep(2)

# Step 1 — Launch controller
print("[1] Launching controller...")
controller_proc = launch_uvicorn("server:app", 8000, "controller")
time.sleep(2)

# # Step 2 — Launch workers
# print("[2] Launching workers...")
# worker_procs = []
# for i in range(len(worker_nodes)):
#     port = 9000 + i
#     proc = launch_uvicorn("client:app", port, "worker")
#     worker_procs.append(proc)
# time.sleep(2)

# Step 3 — Launch dashboard (serve frontend)
print("[3] Launching monitoring dashboard...")
dashboard_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dashboard_proc = subprocess.Popen(['python3', '-m', 'http.server', '8600'], cwd=dashboard_root)
webbrowser.open("http://localhost:8600/ml_dashboard.html")
time.sleep(2)

# # Step 4 — Register iPhone and Raspberry Pi
print("[4] Registering user endpoints to controller...")
devices = [
    # {
    #     "id": "ios-13",
    #     "role": "dashboard",
    #     "hardware": {"ram": 4096, "gpu": False, "cpu": "Apple A15", "device": "ios-13"},
    #     "tags": ["mobile", "observer"]
    # },
    {
        "id": "pi-edge1",
        "role": "inference",
        "hardware": {"ram": 1024, "gpu": False, "cpu": "ARM Cortex-A53", "device": "pi-edge1"},
        "tags": ["low-power", "edge"]
    }
]

for device in devices:
    try:
        res = requests.post(f"{controller_url}/register", json=device)
        print(f"→ Registered {device['id']}: {res.status_code}")
    except Exception as e:
        print(f"✗ Failed to register {device['id']}: {e}")

# Step 5 — Trigger distributed training
print("[5] Sending training tasks to controller...")
try:
    response = requests.post(f"{controller_url}/distribute")
    results = response.json()
    print("→ Controller response:", results)
except Exception as e:
    print("✗ Error distributing training:", e)
    results = {"results": []}

# Step 6 — Wait for aggregation
print("[6] Waiting for aggregation...")
time.sleep(5)

# Step 7 — Optional: Send inference result to Pi (mock)
print("[7] Sending result to Pi-edge1 (simulated endpoint)...")
try:
    final_payload = results["results"][0] if results["results"] else {"error": "No results"}
    dummy_inference_url = "http://localhost:8700/receive"  # Replace with real if needed
    res = requests.post(dummy_inference_url, json=final_payload)
    print("→ Pi-edge1 response:", res.json())
except Exception as e:
    print("✗ Could not contact Pi-edge1:", e)

# Step 8 — Hold dashboard open
print("✓ Simulation complete. Dashboard available for 25 minutes.")
time.sleep(60 * 25)

# Cleanup
print("Shutting down processes...")
# for proc in worker_procs:
#     proc.terminate()
controller_proc.terminate()
# receiver_proc.terminate()
dashboard_proc.terminate()
