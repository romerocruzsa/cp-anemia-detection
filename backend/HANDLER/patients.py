from fastapi.responses import JSONResponse
from datetime import datetime
from DAO.patients import PatientsDAO

class PatientsHandler:
    def mapToDict(self, t):
        return {
            'PatientID': t["patientid"],
            'FirstName': t["firstname"],
            'LastName': t["lastname"],
            'DateOfBirth': t["dateofbirth"].isoformat() if t["dateofbirth"] else None,
            'Gender': t["gender"],
            'Email': t["email"],
            'CreatedAt': t["createdat"].isoformat() if t["createdat"] else None
        }

    async def getAllPatients(self):
        dao = PatientsDAO()
        try:
            dbtuples = await dao.getPatients()
            result = [self.mapToDict(e) for e in dbtuples] if dbtuples else []
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(content={'error': f'An error occurred while retrieving patients: {e}'}, status_code=500)

    async def getPatientsByID(self, pid):
        dao = PatientsDAO()
        try:
            result = await dao.getPatientsById(pid)
            if result:
                return JSONResponse(content=self.mapToDict(result))
            else:
                return JSONResponse(content={"error": "Not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(content={'error': f'Error retrieving patient {pid}: {e}'}, status_code=500)

    async def insertPatient(self, data):
        required_fields = ['FirstName', 'LastName', 'DateOfBirth', 'Gender', 'Email']
        if all(field in data for field in required_fields):
            dao = PatientsDAO()
            try:
                date_of_birth = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
                patient_id = await dao.insertPatients(
                    data['FirstName'],
                    data['LastName'],
                    date_of_birth,  
                    data['Gender'],
                    data['Email']
                )
                return {"PatientID": patient_id, **data}
            except Exception as e:
                raise JSONResponse(status_code=500, detail=f"Error inserting patient: {e}")
        else:
            raise JSONResponse(status_code=400, detail="Bad data or missing fields")

    async def deleteById(self, pid):
        dao = PatientsDAO()
        try:
            result = await dao.deletePatientsById(pid)
            if result:
                return JSONResponse(content={"message": "Delete was successful"}, status_code=200)
            else:
                return JSONResponse(content={"error": "Not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(content={'error': f'Error deleting patient {pid}: {e}'}, status_code=500)

    async def putByID(self, pid, data):
        required_fields = ['FirstName', 'LastName', 'DateOfBirth', 'Gender', 'Email']
        if all(field in data for field in required_fields):
            dao = PatientsDAO()
            try:
                date_of_birth = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
                result = await dao.putPatientsByID(
                    pid,
                    data['FirstName'],
                    data['LastName'],
                    date_of_birth,
                    data['Gender'],
                    data['Email'],
                )
                if result:
                    return JSONResponse(content=data, status_code=200)
                else:
                    return JSONResponse(content={"error": "Not found"}, status_code=404)
            except Exception as e:
                return JSONResponse(content={'error': f'Error updating patient {pid}: {e}'}, status_code=500)
