from fastapi.responses import JSONResponse
from DAO.patients import PatientsDAO

class PatientsHandler:
    def mapToDict(self, t):
        return {
            'PatientID': t["PatientID"],
            'FirstName': t["FirstName"],
            'LastName': t["LastName"],
            'DateOfBirth': t["DateOfBirth"],
            'Gender': t["Gender"],
            'Email': t["Email"],
            'CreatedAt': t["CreatedAt"]
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
        required_fields = ['FirstName', 'LastName', 'DateOfBirth', 'Gender', 'Email', 'CreatedAt']
        if all(field in data for field in required_fields):
            dao = PatientsDAO()
            try:
                patient_id = await dao.insertPatients(None, data['FirstName'], data['LastName'], data['DateOfBirth'], data['Gender'], data['Email'], data['CreatedAt'])
                data['PatientID'] = patient_id
                return JSONResponse(content=data, status_code=201)
            except Exception as e:
                return JSONResponse(content={'error': f'Error inserting patient: {e}'}, status_code=500)
        else:
            return JSONResponse(content={"error": "Bad data or missing fields"}, status_code=400)

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
        required_fields = ['FirstName', 'LastName', 'DateOfBirth', 'Gender', 'Email', 'CreatedAt']
        if all(field in data for field in required_fields):
            dao = PatientsDAO()
            try:
                result = await dao.putPatientsByID(pid, data['FirstName'], data['LastName'], data['DateOfBirth'], data['Gender'], data['Email'], data['CreatedAt'])
                if result:
                    return JSONResponse(content=data, status_code=200)
                else:
                    return JSONResponse(content={"error": "Not found"}, status_code=404)
            except Exception as e:
                return JSONResponse(content={'error': f'Error updating patient {pid}: {e}'}, status_code=500)
        else:
            return JSONResponse(content={"error": "Bad data or missing fields"}, status_code=400)
