
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from HANDLER.images import DataHandler
from HANDLER.patients import PatientsHandler
from backend.MODELS.screening_report import ScreeningReport  

app = FastAPI()

# Pydantic model for request body validation
class DataRequest(BaseModel):
    json_file_path: str

# @app.on_event("shutdown")
# async def shutdown_event():
#     if db_pool:
#         await db_pool.close()

# Handlers
patient_handler = PatientsHandler()
screening_report_handler = ScreeningReportHandler()

# Pydantic model for request body validation
class DataRequest(BaseModel):
    json_file_path: str

@app.post("/add_data")
async def add_data(request: DataRequest):
    json_file_path = request.json_file_path

    if json_file_path:
        data_handler = DataHandler(json_file_path)  # Create an instance of DataHandler
        result = await data_handler.load_and_insert_data()
        return result  # FastAPI automatically converts to JSON
    else:
        raise HTTPException(status_code=400, detail="Missing JSON file path!")

@app.get("/get_data")
async def get_data():
    data_handler = DataHandler('path_to_your_json_file.json')  # Example of creating an instance
    data_handler.fetch_data()
    return {"message": "Check the server logs for fetched data."}

@app.get("/get_patients")
async def get_patients():
    patients = await patient_handler.getAllPatients()
    return patients

@app.post("/screening_reports")
async def create_screening_report(screening_report: ScreeningReport):
    report = await screening_report_handler.create_screening_report(screening_report)
    return report

@app.get("/screening_reports/{screening_report_id}")
async def get_screening_report(screening_report_id: str):
    report = await screening_report_handler.get_screening_report_by_id(screening_report_id)
    if report:
        return report
    raise HTTPException(status_code=404, detail="Screening Report not found")