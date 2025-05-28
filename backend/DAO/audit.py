from CONFIG.db_config import pg_config
import asyncpg
import logging
import json

logging.basicConfig(level=logging.ERROR)

class AuditDAO:
    def __init__(self):
        self.db_config = pg_config
        self.pool = None

    async def connect(self):
        if self.pool is None:
            try:
                self.pool = await asyncpg.create_pool(
                    host=self.db_config['host'],
                    database=self.db_config['dbname'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    port=self.db_config['port']
                )
            except Exception as e:
                logging.error(f"Error creating connection pool: {e}")
                raise

    async def create_audit_record(self, user_id: int, action: str, table_name: str, 
                                record_id: int, old_value: dict = None, new_value: dict = None,
                                ip_address: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO AuditRecords 
                    (UserID, Action, TableName, RecordID, OldValue, NewValue, IPAddress)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    RETURNING AuditID;
                """
                return await conn.fetchval(
                    query, user_id, action, table_name, record_id,
                    json.dumps(old_value) if old_value else None,
                    json.dumps(new_value) if new_value else None,
                    ip_address
                )
            except Exception as e:
                logging.error(f"Error creating audit record: {e}")
                return None

    async def get_audit_records(self, user_id: int = None, table_name: str = None, 
                              start_date: str = None, end_date: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                conditions = []
                params = []
                param_count = 1

                if user_id:
                    conditions.append(f"UserID = ${param_count}")
                    params.append(user_id)
                    param_count += 1

                if table_name:
                    conditions.append(f"TableName = ${param_count}")
                    params.append(table_name)
                    param_count += 1

                if start_date:
                    conditions.append(f"Timestamp >= ${param_count}")
                    params.append(start_date)
                    param_count += 1

                if end_date:
                    conditions.append(f"Timestamp <= ${param_count}")
                    params.append(end_date)
                    param_count += 1

                where_clause = " AND ".join(conditions) if conditions else "1=1"
                query = f"""
                    SELECT a.*, u.Email
                    FROM AuditRecords a
                    JOIN Users u ON a.UserID = u.UserID
                    WHERE {where_clause}
                    ORDER BY a.Timestamp DESC;
                """
                return await conn.fetch(query, *params)
            except Exception as e:
                logging.error(f"Error getting audit records: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}") 