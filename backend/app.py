import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict
from fastapi import Body, FastAPI, HTTPException
from HANDLER.patients import PatientsHandler
from HANDLER.ImageUploads import ImageUploadsHandler
from HANDLER.AnemiaAnalysis import AnemiaAnalysisHandler
from fastapi import UploadFile, File, HTTPException
from HANDLER.HemoglobinEstimator import HemoglobinHandler
import asyncpg
import logging
from fastapi.middleware.cors import CORSMiddleware

handler = PatientsHandler()
image_handler = ImageUploadsHandler()
analysis_handler = AnemiaAnalysisHandler()
hemoglobin_handler = HemoglobinHandler()
app = FastAPI()

# 👇 Add this after `app = FastAPI()`
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only — allows all domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
async def insert_patient(data: Dict = Body(...)):
    try:
        result = await handler.insertPatient(data)
        if isinstance(result, dict):
            return result
        return result.body
    except Exception as e:
        logging.error(f"Error in insert_patient: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_patient/{pid}")
async def delete_patient(pid: int):
    return await handler.deleteById(pid)

@app.put("/update_patient/{pid}")
async def update_patient(pid: int, data: dict):
    return await handler.putByID(pid, data)

@app.get("/get_uploads")
async def get_uploads():
    return await image_handler.getAllUploads()

@app.post("/create_upload/{patient_id}")
async def upload_image(patient_id: int, image_path: str):
    return await image_handler.createUpload(patient_id, image_path)

@app.get("/get_uploads_status/{patient_id}")
async def get_patient_uploads(patient_id: int):
    return await image_handler.getUploadsByPatient(patient_id)

@app.put("/update_upload_status/{image_id}/status")
async def update_upload_status(image_id: int, status: str):
    return await image_handler.updateUploadStatus(image_id, status)

@app.put("/update_image/{image_id}")
async def update_image(image_id: int, image_path: str):
    return await image_handler.updateImage(image_id, image_path)

@app.delete("/delete_image/{image_id}")
async def delete_image(image_id: int):
    return await image_handler.deleteImage(image_id)

@app.get("/get_Analysis")
async def get_uploads():
    return await analysis_handler.getAnalysis()

@app.post("/create_analysis/{image_id}")
async def create_analysis(image_id: int, status: str, confidence: float):
    return await analysis_handler.createAnalysis(image_id, status, confidence)

@app.get("/get_analysis_by_id/{image_id}")
async def get_analysis(image_id: int):
    return await analysis_handler.getAnalysisByImage(image_id)

@app.get("/analysis/patient/{patient_id}")
async def get_patient_history(patient_id: int):
    return await analysis_handler.getPatientHistory(patient_id)

@app.delete("/delete_analysis/{analysis_id}")
async def delete_analysis(analysis_id: int):
    return await analysis_handler.deleteAnalysis(analysis_id)

@app.put("/update_analysis/{analysis_id}")
async def update_analysis(analysis_id: int, status: str, confidence: float):
    return await analysis_handler.updateAnalysis(analysis_id, status, confidence)

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), patient_id: int = None, image_id: int = None):
    try:
        image_bytes = await file.read()
        async with asyncpg.create_pool(
            host="localhost",
            database="capiku",
            user="capiku",
            password="capiku@3131!",
            port="5433"
        ) as pool:
            async with pool.acquire() as conn:
                result = hemoglobin_handler.predict_hgb(image_bytes)
                # print(result)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 This will show full traceback in terminal
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
