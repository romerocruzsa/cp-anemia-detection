from fastapi import FastAPI
import random
import time

app = FastAPI()

@app.post("/train")
def train_model():
    time.sleep(2)
    return {
        "accuracy": round(random.uniform(0.85, 0.9), 4),
        "loss": round(random.uniform(0.1, 0.2), 4),
        "latency": round(random.uniform(120, 180), 2),  # in ms
        "power": round(random.uniform(2.5, 3.5), 2),     # in watts
        "ram": round(random.uniform(128, 256), 2),       # in MB
        "storage": round(random.uniform(40, 60), 2),      # in MB
        "aurora": round(random.uniform(2.25, 3.45), 4)
    }