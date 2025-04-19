from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from HANDLER.images import DataHandler
from HANDLER.patients import PatientsHandler  
from ROUTES import auth
from ROUTES.auth import get_current_user, require_role
from audit import log_action

app = FastAPI()
app.include_router(auth.router) # Include auth routes (register, token, role check)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")  # Matches the auth router path

# Pydantic model for data input
class DataRequest(BaseModel):
    json_file_path: str

@app.post("/add_data")
async def add_data(request: DataRequest, current_user=Depends(require_role("clinician"))):
    json_file_path = request.json_file_path

    if json_file_path:
        data_handler = DataHandler(json_file_path)
        result = await data_handler.load_and_insert_data()
        return result
    else:
        raise HTTPException(status_code=400, detail="Missing JSON file path!")

@app.get("/get_data")
async def get_data(current_user=Depends(require_role("clinician"))):
    data_handler = DataHandler('path_to_your_json_file.json')
    data_handler.fetch_data()
    return {"message": "Check the server logs for fetched data."}

@app.get("/patients/")
async def get_all_patients(current_user=Depends(require_role("clinician"))):
    await log_action(
        user_email=current_user["sub"],
        action="READ_PATIENTS",
        resource="patients"
    )
    handler = PatientsHandler()
    return handler.getAllPatients()

# Future implementation
# @app.post("/patients/")
# async def add_patient(current_user=Depends(require_role("clinician"))):
#     await log_action(
#         user_email=current_user["sub"],
#         action="READ_PATIENTS",
#         resource="patients"
#     )
#     handler = PatientsHandler()
#     return handler.createPatient()
