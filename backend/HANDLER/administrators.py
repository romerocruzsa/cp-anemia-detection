from fastapi import HTTPException
from fastapi.responses import JSONResponse
from DAO.administrators import AdministratorsDAO
import logging

logging.basicConfig(level=logging.ERROR)

class AdministratorsHandler:
    def __init__(self):
        self.dao = AdministratorsDAO()

    def mapToDict(self, row: dict):
        return {
            'AdminID': row["adminid"],
            'UserID': row["userid"],
            'FirstName': row["firstname"],
            'LastName': row["lastname"],
            'Email': row["email"],
            'Role': row["role"],
            'CreatedAt': row["createdat"].isoformat() if row["createdat"] else None
        }

    async def getAdministratorByID(self, aid: int):
        try:
            row = await self.dao.get_administrator_by_id(aid)
            if not row:
                raise HTTPException(status_code=404, detail='Administrator not found')
            return self.mapToDict(row)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving administrator {aid}: {e}')

    async def getAdministratorByUserID(self, user_id: int):
        try:
            row = await self.dao.get_administrator_by_user_id(user_id)
            if not row:
                raise HTTPException(status_code=404, detail='Administrator not found')
            return self.mapToDict(row)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving administrator for user {user_id}: {e}')

    async def getAllAdministrators(self):
        try:
            rows = await self.dao.get_all_administrators()
            result = [self.mapToDict(r) for r in rows] if rows else []
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Error retrieving administrators: {e}') 