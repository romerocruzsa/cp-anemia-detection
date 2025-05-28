import sys
import os
import time
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import logging

# ── Load .env from CONFIG/local.env ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join("CONFIG", "local.env"))

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from HANDLER.patients import PatientsHandler
from HANDLER.ImageUploads import ImageUploadsHandler
from HANDLER.AnemiaAnalysis import AnemiaAnalysisHandler
from HANDLER.HemoglobinEstimator import HemoglobinHandler
from HANDLER.audit import AuditHandler
from HANDLER.medical_notes import MedicalNotesHandler
from HANDLER.auth import AuthHandler
from db.database import get_connection, release_connection

# Global handler instances
handler = PatientsHandler()
image_handler = ImageUploadsHandler()
analysis_handler = AnemiaAnalysisHandler()
hgb_handler = None  # Loaded via lifespan
audit_handler = AuditHandler()
notes_handler = MedicalNotesHandler()
auth_handler = AuthHandler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hgb_handler
    hgb_handler = HemoglobinHandler()
    print("✅ Hemoglobin models loaded at startup!")
    yield

app = FastAPI(lifespan=lifespan)

# CORS middleware (you can restrict this for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test_db")
async def test_db():
    """
    Quick raw check that database connection is working.
    """
    try:
        conn = await get_connection()
        try:
            result = await conn.fetch("SELECT * FROM Patients;")
            return {"patients": result}
        finally:
            await release_connection(conn)
    except Exception as e:
        logging.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to database")

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

@app.post("/register")
async def register(request: Request):
    try:
        data = await request.json()
        return await auth_handler.register(data)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    except Exception as e:
        logging.error(f"Registration error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.post("/login")
async def login(request: Request):
    try:
        data = await request.json()
        email = data.get('Email')
        password = data.get('Password')
        
        if not email or not password:
            return JSONResponse(
                status_code=400,
                content={"detail": "Email and password are required"}
            )
            
        return await auth_handler.login(email, password)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail}
        )
    except Exception as e:
        logging.error(f"Login error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...), patient_id: int = None, image_id: int = None):
    try:
        start_time = time.time()
        print("✅ Starting Prediction...")
        image_bytes = await file.read()
        result = hgb_handler(image_bytes)
        elapsed_time = time.time() - start_time
        print(f"✅ Estimation Complete! Took ~{elapsed_time:.2f} seconds.")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Audit endpoints
@app.get("/audit/trail")
async def get_audit_trail(user_id: int = None, table_name: str = None,
                         start_date: str = None, end_date: str = None):
    try:
        records = await audit_handler.get_audit_trail(user_id, table_name, start_date, end_date)
        return JSONResponse(content=records)
    except Exception as e:
        logging.error(f"Error retrieving audit trail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Medical Notes endpoints
@app.post("/notes/create")
async def create_medical_note(data: Dict = Body(...)):
    try:
        note_id = await notes_handler.create_note(
            data.get("clinician_id"),
            data.get("patient_id"),
            data.get("analysis_id"),
            data.get("note_text"),
            data.get("follow_up_date")
        )
        if note_id:
            return JSONResponse(content={"note_id": note_id, "message": "Note created successfully"})
        raise HTTPException(status_code=500, detail="Failed to create note")
    except Exception as e:
        logging.error(f"Error creating medical note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notes/patient/{patient_id}")
async def get_patient_notes(patient_id: int):
    try:
        notes = await notes_handler.get_patient_notes(patient_id)
        return JSONResponse(content=notes)
    except Exception as e:
        logging.error(f"Error retrieving patient notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notes/{note_id}")
async def get_note(note_id: int):
    try:
        note = await notes_handler.get_note_by_id(note_id)
        if note:
            return JSONResponse(content=note)
        raise HTTPException(status_code=404, detail="Note not found")
    except Exception as e:
        logging.error(f"Error retrieving note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/notes/{note_id}")
async def update_note(note_id: int, data: Dict = Body(...)):
    try:
        updated_id = await notes_handler.update_note(
            note_id,
            data.get("note_text"),
            data.get("follow_up_date")
        )
        if updated_id:
            return JSONResponse(content={"message": "Note updated successfully"})
        raise HTTPException(status_code=404, detail="Note not found")
    except Exception as e:
        logging.error(f"Error updating note: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/notes/follow-ups")
async def get_pending_follow_ups(clinician_id: int = None):
    try:
        follow_ups = await notes_handler.get_pending_follow_ups(clinician_id)
        return JSONResponse(content=follow_ups)
    except Exception as e:
        logging.error(f"Error retrieving follow-ups: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
