from fastapi import FastAPI, HTTPException
from HANDLER.patients import PatientsHandler  

app = FastAPI()


@app.get("/get_patients")
async def get_patients():
    return PatientsHandler().getAllPatients()


