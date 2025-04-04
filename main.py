import subprocess
import time
import webbrowser
import os
import sys

def launch_backend():
    print("[1] Launching controller API...")
    return subprocess.Popen(['uvicorn', 'server:app', '--port', '8000', '--reload'], cwd="distributed_training/controller")

def launch_dashboard():
    print("[2] Launching monitoring dashboard...")
    dashboard_root = os.path.abspath(os.path.dirname(__file__))
    webbrowser.open("http://localhost:5500/ml_dashboard.html")
    return subprocess.Popen(["python3", "-m", "http.server", "5500"], cwd=dashboard_root)

def main():
    api = launch_backend()
    time.sleep(2)
    dash = launch_dashboard()
    print("[✓] Dashboard and controller ready. Configure training in your browser.")

    try:
        # Keep script alive while dashboard is up
        dash.wait()
    except KeyboardInterrupt:
        print("\n[✗] Shutdown requested by user.")

    print("[!] Shutting down processes...")
    api.terminate()
    dash.terminate()

if __name__ == "__main__":
    main()
