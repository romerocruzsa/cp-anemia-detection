from fastapi import HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from DAO.users import UsersDAO
from DAO.patients import PatientsDAO
from DAO.clinicians import CliniciansDAO
import logging

logging.basicConfig(level=logging.ERROR)

class AuthHandler:
    def __init__(self):
        self.users_dao = UsersDAO()
        self.patients_dao = PatientsDAO()
        self.clinicians_dao = CliniciansDAO()

    async def register(self, data: dict):
        required_fields = {
            'patient': ['FirstName', 'LastName', 'DateOfBirth', 'Gender', 'Email', 'Password'],
            'clinician': ['FirstName', 'LastName', 'Email', 'Password', 'LicenseNumber', 'Specialization'],
            'administrator': ['FirstName', 'LastName', 'Email', 'Password', 'AdminCode']
        }

        role = data.get('Role')
        if not role or role not in required_fields:
            raise HTTPException(status_code=400, detail="Invalid role")

        # Check required fields
        if not all(field in data for field in required_fields[role]):
            raise HTTPException(status_code=400, detail=f"Missing required fields for {role}")

        # Validate admin code if registering as administrator
        if role == 'administrator':
            if data['AdminCode'] != 'ADMIN123':  # Replace with secure admin code validation
                raise HTTPException(status_code=401, detail="Invalid administrator code")

        try:
            # Create user record first
            user_id = await self.users_dao.create_user(
                data['Email'],
                data['Password'],
                role
            )

            if not user_id:
                raise HTTPException(status_code=500, detail="Failed to create user account")

            # Create role-specific record
            if role == 'patient':
                record_id = await self.patients_dao.insertPatientWithPassword(
                    user_id,
                    data['FirstName'],
                    data['LastName'],
                    data['DateOfBirth'],
                    data['Gender'],
                    data['Email'],
                    data['Password']
                )
                if not record_id:
                    raise HTTPException(status_code=500, detail="Failed to create patient record")
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'PatientID': record_id,
                        'Role': role,
                        'message': 'Registration successful'
                    }
                )
            elif role == 'clinician':
                record_id = await self.clinicians_dao.create_clinician(
                    user_id,
                    data['FirstName'],
                    data['LastName'],
                    data['LicenseNumber'],
                    data['Specialization']
                )
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'ClinicianID': record_id,
                        'Role': role,
                        'message': 'Registration successful'
                    }
                )
            elif role == 'administrator':
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'AdminID': user_id,  # For admins, we use the UserID as AdminID
                        'Role': role,
                        'message': 'Registration successful'
                    }
                )

        except Exception as e:
            logging.error(f"Registration error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def login(self, email: str, password: str):
        try:
            # Validate credentials and get user info
            user_id, role = await self.users_dao.validate_login(email, password)
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid email or password")

            # Get role-specific ID
            if role == 'patient':
                patient = await self.patients_dao.getPatientsByUserId(user_id)
                if not patient:
                    raise HTTPException(status_code=404, detail="Patient record not found")
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'PatientID': patient['patientid'],
                        'Role': role,
                        'message': 'Login successful'
                    }
                )
            elif role == 'clinician':
                clinician = await self.clinicians_dao.get_clinician_by_user_id(user_id)
                if not clinician:
                    raise HTTPException(status_code=404, detail="Clinician record not found")
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'ClinicianID': clinician['clinicianid'],
                        'Role': role,
                        'message': 'Login successful'
                    }
                )
            elif role == 'administrator':
                return JSONResponse(
                    status_code=200,
                    content={
                        'UserID': user_id,
                        'AdminID': user_id,
                        'Role': role,
                        'message': 'Login successful'
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid user role")

        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"Login error: {e}")
            raise HTTPException(status_code=500, detail=str(e)) 