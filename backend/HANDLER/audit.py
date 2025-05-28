from DAO.audit import AuditDAO
import logging
from datetime import datetime

logging.basicConfig(level=logging.ERROR)

class AuditHandler:
    def __init__(self):
        self.dao = AuditDAO()

    async def log_action(self, user_id: int, action: str, table_name: str, 
                        record_id: int, old_value: dict = None, new_value: dict = None,
                        ip_address: str = None):
        try:
            audit_id = await self.dao.create_audit_record(
                user_id, action, table_name, record_id, old_value, new_value, ip_address
            )
            return audit_id
        except Exception as e:
            logging.error(f"Error logging action: {e}")
            return None

    async def get_audit_trail(self, user_id: int = None, table_name: str = None,
                            start_date: str = None, end_date: str = None):
        try:
            records = await self.dao.get_audit_records(user_id, table_name, start_date, end_date)
            if records:
                return [dict(record) for record in records]
            return []
        except Exception as e:
            logging.error(f"Error getting audit trail: {e}")
            return []

    async def close(self):
        await self.dao.close_connection() 