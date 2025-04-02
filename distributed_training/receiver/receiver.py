# result_receiver/receiver.py
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/receive")
async def receive_result(request: Request):
    data = await request.json()
    print("Final result received by receiver:", data)
    return {"status": "received"}
