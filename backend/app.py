from fastapi import FastAPI, HTTPException
from HANDLER.patients import PatientsHandler
import asyncpg
import logging


handler = PatientsHandler()
app = FastAPI()


@app.get("/test_db")
async def test_db():
    try:
        async with asyncpg.create_pool(
            host="localhost",
            database="capiku",
            user="capiku",
            password="capiku@3131!",
            port="5433"
        ) as pool:
            async with pool.acquire() as conn:
                result = await conn.fetch("SELECT * FROM Patients;")
                return {"patients": result}
    except Exception as e:
        logging.error(f"Database error: {e}")
        return {"error": "Failed to connect to database"}, 500


@app.get("/get_patients")
async def get_patients():
    try:
        return await handler.getAllPatients()
    except Exception as e:
        logging.error(f"Error retrieving patients: {e}")
        return {"error": "Failed to retrieve patients"}, 500

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


