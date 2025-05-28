from fastapi import HTTPException
from fastapi.responses import JSONResponse
from DAO.clinicians import CliniciansDAO
import logging

logging.basicConfig(level=logging.ERROR)

class CliniciansHandler:
    def mapToDict(self, row: dict):
        logging.error(f"Mapping row: {row}")  # Debug log
        mapped = {
            'ClinicianID': row["clinicianid"],
            'FirstName': row["firstname"],
            'LastName': row["lastname"],
            'LicenseNumber': row["licensenumber"],
            'Specialization': row["specialization"],
            'Email': row["email"],
            'CreatedAt': row["createdat"].isoformat() if row["createdat"] else None
        }
        logging.error(f"Mapped result: {mapped}")  # Debug log
        return mapped

    async def getClinicianByID(self, cid: int):
        dao = CliniciansDAO()
        try:
            row = await dao.get_clinician_by_id(cid)
            logging.error(f"Got row from DAO: {row}")  # Debug log
            if not row:
                raise HTTPException(status_code=404, detail='Clinician not found')
            result = self.mapToDict(row)
            logging.error(f"Final result: {result}")  # Debug log
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving clinician {cid}: {e}')

    async def getClinicianByUserID(self, user_id: int):
        dao = CliniciansDAO()
        try:
            row = await dao.get_clinician_by_user_id(user_id)
            logging.error(f"Got row from DAO by user_id: {row}")  # Debug log
            if not row:
                raise HTTPException(status_code=404, detail='Clinician not found')
            result = self.mapToDict(row)
            logging.error(f"Final result by user_id: {result}")  # Debug log
            return result
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving clinician for user {user_id}: {e}')

    async def getAllClinicians(self):
        dao = CliniciansDAO()
        try:
            rows = await dao.get_all_clinicians()
            result = [self.mapToDict(r) for r in rows] if rows else []
            logging.error(f"All clinicians result: {result}")  # Debug log
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving clinicians: {e}') 