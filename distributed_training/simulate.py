import os
import subprocess
import time import
import requests
import json
import webbrowser
from main import main as train_main  # Import your training function

def launch_uvicorn(app_path, port, cwd):
    return subprocess.Popen(['uvicorn', app_path, '--port', str(port), '--reload'], cwd=cwd)

def load_config():    
    config_path = os.path.join(os.path.dirname(__file__), "configs/nodes.json")
    with open(config_path) as f:
        return json.load(f)

def run_main_with_config(config_dict):
    config_path = os.path.join(os.path.dirname(__file__), "..", "training_config.json")

    # Save the config to a temporary file in root
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    # Absolute path to main.py from simulate.py
    main_py_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))

    # Call main.py as subprocess with the config path
    subprocess.run(["python", main_py_path, config_path])

# Setup
config = load_config()
controller_url = config["controller"]
receiver_url = config["receiver"]
controller_id = "controller"

# Step 1 — Launch controller
print("[1] Launching controller...")
controller_proc = launch_uvicorn("server:app", 8000, "controller")
time.sleep(2)

# Step 2 — Launch dashboard
print("[2] Launching monitoring dashboard...")
dashboard_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dashboard_proc = subprocess.Popen(['python3', '-m', 'http.server', '5500'], cwd=dashboard_root)
time.sleep(2)

# Step 3 — Register edge devices
print("[3] Registering user endpoints to controller...")
devices = [
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

# Step 4 — Trigger distributed training config
print("[4] Sending training tasks to controller...")
training_config = {
    "learningRate": 0.001,
    "batchSize": 32,
    "epochs": 1,
    "selectedRoles": [{"id": controller_id, "role": "training"}]
}

try:
    response = requests.post(f"{controller_url}/start-training", json=training_config)
    print("→ Controller response:", response.json())
except Exception as e:
    print("✗ Error contacting controller:", e)

# Step 5 — If controller is trainer, run training
if any(role["id"] == controller_id and role["role"] == "training" for role in training_config["selectedRoles"]):
    print(f"[✓] Local node '{controller_id}' is a trainer. Running local training...")
    result = train_main(training_config)
    print("[✓] Training result:", result)

# # Step 6 — Simulate inference response
# print("[6] Sending result to Pi-edge1 (simulated endpoint)...")
# try:
#     dummy_result = result if result else {"error": "No result"}
#     dummy_inference_url = "http://localhost:8700/receive"
#     res = requests.post(dummy_inference_url, json=dummy_result)
#     print("→ Pi-edge1 response:", res.json())
# except Exception as e:
#     print("✗ Could not contact Pi-edge1:", e)

print("✓ Simulation complete. Dashboard available for 25 minutes.")
time.sleep(60 * 25)

# Cleanup
print("Shutting down processes...")
controller_proc.terminate()
dashboard_proc.terminate()
