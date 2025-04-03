from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/receive")
async def receive_results(req: Request):
    data = await req.json()
    print("Received final result:", data)
    with open("jobs/results/final_output.json", "w") as f:
        json.dump(data, f)
    return {"status": "received"}
