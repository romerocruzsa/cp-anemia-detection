from fastapi import HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from DAO.patients import PatientsDAO
from DAO.users import UsersDAO

class PatientsHandler:
    def __init__(self):
        self.dao = PatientsDAO()
        self.users_dao = UsersDAO()

    def mapToDict(self, patient):
        if patient is None:
            return None
        return {
            "PatientID": patient['patientid'],
            "UserID": patient['userid'],
            "FirstName": patient['firstname'],
            "LastName": patient['lastname'],
            "DateOfBirth": patient['dateofbirth'].isoformat() if patient['dateofbirth'] else None,
            "Gender": patient['gender'],
            "Email": patient['email'] if 'email' in patient else None,
            "Role": patient['role'] if 'role' in patient else None,
            "CreatedAt": patient['createdat'].isoformat() if patient['createdat'] else None,
            "UpdatedAt": patient['updatedat'].isoformat() if patient['updatedat'] else None
        }

    async def getPatientByID(self, pid):
        patient = await self.dao.getPatientsById(pid)
        return self.mapToDict(patient)

    async def getPatientByUserID(self, user_id):
        patient = await self.dao.getPatientsByUserId(user_id)
        return self.mapToDict(patient)

    async def getPatientsByID(self, pid: int):
        try:
            row = await self.dao.getPatientsById(pid)
            if not row:
                raise HTTPException(status_code=404, detail='Patient not found')
            return self.mapToDict(row)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving patient {pid}: {e}')

    async def getAllPatients(self):
        patients = await self.dao.getPatients()
        return [self.mapToDict(patient) for patient in patients]

    def insertPatientWithPassword(self, user_id, first_name, last_name, date_of_birth, gender, email, password):
        try:
            # Convert date string to date object if it's a string
            if isinstance(date_of_birth, str):
                date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            
            return self.dao.insert_patient_with_password(
                user_id=user_id,
                first_name=first_name,
                last_name=last_name,
                date_of_birth=date_of_birth,
                gender=gender,
                email=email,
                password=password
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def insertPatientWithPassword(self, data: dict):
        required = ['FirstName','LastName','DateOfBirth','Gender','Email','Password']
        if not all(k in data for k in required):
            raise HTTPException(status_code=400, detail="Missing required fields")

        try:
            dob = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="DateOfBirth must be YYYY-MM-DD")

        try:
            patient_id = await self.dao.insert_patient_with_password(
                data['FirstName'],
                data['LastName'],
                dob,
                data['Gender'],
                data['Email'],
                data['Password']
            )
            if not patient_id:
                raise HTTPException(status_code=500, detail='Could not register patient')
            return {'PatientID': patient_id, 'message': 'Registration successful'}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error inserting patient: {e}')

    async def loginPatient(self, email: str, password: str):
        pid = await self.dao.validate_patient_login(email, password)
        return pid

    async def putByID(self, pid: int, data: dict):
        required = ['FirstName','LastName','DateOfBirth','Gender','Email']
        if not all(k in data for k in required):
            raise HTTPException(status_code=400, detail="Missing required fields")

        try:
            dob = datetime.strptime(data['DateOfBirth'], "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="DateOfBirth must be YYYY-MM-DD")

        try:
            result = await self.dao.put_patients_by_id(
                pid,
                data['FirstName'],
                data['LastName'],
                dob,
                data['Gender'],
                data['Email']
            )
            if not result:
                raise HTTPException(status_code=404, detail='Patient not found')
            return data
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error updating patient {pid}: {e}')

    async def deleteById(self, pid: int):
        try:
            result = await self.dao.delete_patients_by_id(pid)
            if not result:
                raise HTTPException(status_code=404, detail='Patient not found')
            return {'message': 'Delete successful'}
        except HTTPException:
            raise
        except Exception as e:
            if result:
                return JSONResponse(status_code=200, content={'message': 'Delete successful'})
            else:
                return JSONResponse(status_code=404, content={'error': 'Not found'})
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={'error': f'Error deleting patient {pid}: {e}'}
            )
