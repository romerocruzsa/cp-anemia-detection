from fastapi.responses import JSONResponse
from DAO.patients import PatientsDAO
from app import get_current_user
class PatientsHandler:

    def mapToDict(self, t):
        result = {
            'PatientID': t[0],
            'FirstName': t[1],
            'LastName': t[2],
            'DateOfBirth': t[3],
            'Gender': t[4],
            'Email': t[5],
            'CreatedAt': t[6]
        }
        return result

    def getAllPatients(self):
        dao = PatientsDAO()
        try:
            dbtuples = dao.getPatients()
            result = [self.mapToDict(e) for e in dbtuples]
            return JSONResponse(content=result)
        except Exception as e:
            print(f"An error occurred while getting all patients: {e}")
            return JSONResponse(content={'error': 'An error occurred while retrieving patients'}, status_code=500)
        
    def createPatient(self, patient_data: dict):
        dao = PatientsDAO()
        try:
            result = dao.createPatient(patient_data)
            return JSONResponse(content=result)
        except Exception as e:
            print(f"Error creating patient: {e}")
            return JSONResponse(content={'error': 'Failed to create patient'}, status_code=500)