from fastapi import FastAPI
import random
import time

app = FastAPI()

@app.post("/train")
def train_model():
    time.sleep(2)
    return {
        "accuracy": round(random.uniform(0.85, 0.9), 4),
        "loss": round(random.uniform(0.1, 0.2), 4)
    }
