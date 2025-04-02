from fastapi import FastAPI, HTTPException
from HANDLER.patients import PatientsHandler

app = FastAPI()
handler = PatientsHandler()

@app.get("/get_patients")
async def get_patients():
    return await handler.getAllPatients()

@app.get("/get_patient/{pid}")
async def get_patient_by_id(pid: int):
    return await handler.getPatientsByID(pid)

@app.post("/insert_patient")
async def insert_patient(data: dict):
    return await handler.insertPatient(data)

@app.delete("/delete_patient/{pid}")
async def delete_patient(pid: int):
    return await handler.deleteById(pid)

@app.put("/update_patient/{pid}")
async def update_patient(pid: int, data: dict):
    return await handler.putByID(pid, data)


