from fastapi import HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from DAO.patients import PatientsDAO

class PatientsHandler:
    def mapToDict(self, row: dict):
        return {
            'PatientID':   row["patientid"],
            'FirstName':   row["firstname"], # already decrypted by DAO
            'LastName':    row["lastname"], # already decrypted by DAO
            'DateOfBirth': row["dateofbirth"].isoformat() if row["dateofbirth"] else None,
            'Gender':      row["gender"],
            'Email':       row["email"], # already decrypted by DAO
            # 'BloodType':   row.get("bloodtype"),    # if you added this column
            # 'Condition':   row.get("condition"),    # and this one
            'CreatedAt':   row["createdat"].isoformat() if row["createdat"] else None
            }

    async def getAllPatients(self):
        dao = PatientsDAO()
        try:
            rows = await dao.getPatients()
            result = [ self.mapToDict(r) for r in rows ] if rows else []
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error retrieving patients: {e}'}
            )

    async def getPatientsByID(self, pid: int):
        dao = PatientsDAO()
        try:
            row = await dao.getPatientsById(pid)
            if not row:
                return JSONResponse(status_code=404, content={'error': 'Not found'})
            return JSONResponse(content=self.mapToDict(row))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error retrieving patient {pid}: {e}'}
            )

    async def insertPatientWithPassword(self, data: dict):
        # This is your registration endpoint
        required = ['FirstName','LastName','DateOfBirth','Gender','Email','Password']
        if not all(k in data for k in required):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # parse the date
        try:
            dob = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="DateOfBirth must be YYYY-MM-DD")

        dao = PatientsDAO()
        try:
            patient_id = await dao.insertPatientWithPassword(
                data['FirstName'],
                data['LastName'],
                dob,
                data['Gender'],
                data['Email'],
                data['Password']
            )
            if not patient_id:
                return JSONResponse(
                    status_code=500,
                    content={'error': 'Could not register patient'}
                )
            return JSONResponse(
                status_code=200,
                content={'PatientID': patient_id, 'message': 'Registration successful'}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error inserting patient: {e}'}
            )

    async def loginPatient(self, email: str, password: str):
        # Simply returns the patient_id or None
        dao = PatientsDAO()
        pid = await dao.validatePatientLogin(email, password)
        return pid

    async def putByID(self, pid: int, data: dict):
        # Updating names, DOB, gender, email (all encrypted/decrypted by DAO)
        required = ['FirstName','LastName','DateOfBirth','Gender','Email']
        if not all(k in data for k in required):
            raise HTTPException(status_code=400, detail="Missing required fields")

        try:
            dob = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="DateOfBirth must be YYYY-MM-DD")

        dao = PatientsDAO()
        try:
            result = await dao.putPatientsByID(
                pid,
                data['FirstName'],
                data['LastName'],
                dob,
                data['Gender'],
                data['Email']
            )
            if result:
                return JSONResponse(status_code=200, content=data)
            else:
                return JSONResponse(status_code=404, content={'error': 'Not found'})
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error updating patient {pid}: {e}'}
            )

    async def deleteById(self, pid: int):
        dao = PatientsDAO()
        try:
            result = await dao.deletePatientsById(pid)
            if result:
                return JSONResponse(status_code=200, content={'message': 'Delete successful'})
            else:
                return JSONResponse(status_code=404, content={'error': 'Not found'})
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error deleting patient {pid}: {e}'}
            )
