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
    distributed_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(distributed_dir, "configs/nodes.json")) as f:
        return json.load(f)

# Load configuration
config = load_config()
controller_url = config['controller']
worker_nodes = config['workers']
receiver_url = config['receiver']

# Step 0 — Launch result receiver
print("[0] Launching result receiver...")
receiver_proc = launch_uvicorn("receiver:app", 8500, "receiver")
time.sleep(2)

# Step 1 — Launch controller
print("[1] Launching controller...")
controller_proc = launch_uvicorn("server:app", 8000, "controller")
time.sleep(2)

# Step 2 — Launch workers
print("[2] Launching workers...")
worker_procs = []
for i in range(len(worker_nodes)):
    port = 9000 + i
    proc = launch_uvicorn("client:app", port, "worker")
    worker_procs.append(proc)
time.sleep(2)

# Step 3 — Serve ml_dashboard.html
print("[3] Launching monitoring dashboard...")
dashboard_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # one level up
dashboard_proc = subprocess.Popen(['python3', '-m', 'http.server', '8600'], cwd=dashboard_dir)
time.sleep(1)
webbrowser.open("http://localhost:8600/ml_dashboard.html")

# Step 4 — Trigger training
print("[4] Sending training tasks to controller...")
try:
    response = requests.post(f"{controller_url}/distribute")
    print("→ Controller response:", response.json())
except Exception as e:
    print("Error contacting controller:", e)

# Step 5 — Wait and simulate aggregation
print("[5] Waiting for aggregation...")
time.sleep(5)

# Step 6 — Send result to receiver
print("[6] Sending final result to receiver...")
final_payload = response.json()["results"][0]
try:
    res = requests.post(f"{receiver_url}/receive", json=final_payload)
    print("→ Receiver response:", res.json())
except Exception as e:
    print("Error contacting receiver:", e)

print("Simulation complete!")

time.sleep(60*5)

# Optional: clean up
for proc in worker_procs:
    proc.terminate()
controller_proc.terminate()
receiver_proc.terminate()
time.sleep(60*2)
dashboard_proc.terminate()
