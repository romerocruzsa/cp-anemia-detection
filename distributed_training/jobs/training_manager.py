import time
import random

def simulate_training(node_id: str) -> dict:
    time.sleep(2)
    history = [round(random.uniform(0.85, 0.9), 4) for _ in range(10)]
    return {
        "id": node_id,
        "status": "complete",
        "accuracy": history[-1],
        "loss": round(random.uniform(0.1, 0.2), 4),
        "latency": round(random.uniform(120, 180), 2),
        "power": round(random.uniform(2.5, 3.5), 2),
        "ram": round(random.uniform(1024, 4096), 2),
        "storage": round(random.uniform(40, 60), 2),
        "aurora": round(random.uniform(2.2, 3.5), 4),
        "history": history
    }

def simulate_inference(node_id: str) -> dict:
    time.sleep(1)
    return {
        "id": node_id,
        "status": "complete",
        "output": random.choice(["positive", "negative"]),
        "confidence": round(random.uniform(0.7, 0.99), 4)
    }
