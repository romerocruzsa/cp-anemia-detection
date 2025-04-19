
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from HANDLER.images import DataHandler
from HANDLER.patients import PatientsHandler  

app = FastAPI()

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
    return PatientsHandler().getAllPatients()

# @app.post("/patients/") # Still needs tweaking!
# async def add_patient():
#     handler = PatientsHandler()
#     return handler.createPatient()

