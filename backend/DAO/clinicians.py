from CONFIG.db_config import pg_config
import asyncpg
import logging

logging.basicConfig(level=logging.ERROR)

class CliniciansDAO:
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

    async def create_clinician(self, user_id: int, first_name: str, last_name: str, 
                             license_number: str, specialization: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO Clinicians (UserID, FirstName, LastName, LicenseNumber, Specialization)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING ClinicianID;
                """
                return await conn.fetchval(query, user_id, first_name, last_name, 
                                        license_number, specialization)
            except Exception as e:
                logging.error(f"Error creating clinician: {e}")
                return None

    async def get_clinician_by_id(self, clinician_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT c.*, u.Email
                    FROM Clinicians c
                    JOIN Users u ON c.UserID = u.UserID
                    WHERE c.ClinicianID = $1;
                """
                result = await conn.fetchrow(query, clinician_id)
                logging.error(f"Raw clinician data: {result}")  # Debug log
                if result:
                    result_dict = dict(result)
                    logging.error(f"Clinician dict: {result_dict}")  # Debug log
                return result
            except Exception as e:
                logging.error(f"Error getting clinician: {e}")
                return None

    async def get_clinician_by_user_id(self, user_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT c.*, u.Email
                    FROM Clinicians c
                    JOIN Users u ON c.UserID = u.UserID
                    WHERE c.UserID = $1;
                """
                result = await conn.fetchrow(query, user_id)
                logging.error(f"Raw clinician data by user_id: {result}")  # Debug log
                if result:
                    result_dict = dict(result)
                    logging.error(f"Clinician dict by user_id: {result_dict}")  # Debug log
                return result
            except Exception as e:
                logging.error(f"Error getting clinician by user ID: {e}")
                return None

    async def update_clinician(self, clinician_id: int, first_name: str, last_name: str, 
                             license_number: str, specialization: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    UPDATE Clinicians
                    SET FirstName = $1, LastName = $2, LicenseNumber = $3, Specialization = $4
                    WHERE ClinicianID = $5;
                """
                return await conn.execute(query, first_name, last_name, 
                                       license_number, specialization, clinician_id)
            except Exception as e:
                logging.error(f"Error updating clinician: {e}")
                return None

    async def get_all_clinicians(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT c.*, u.Email
                    FROM Clinicians c
                    JOIN Users u ON c.UserID = u.UserID
                    WHERE c.IsActive = true;
                """
                return await conn.fetch(query)
            except Exception as e:
                logging.error(f"Error getting all clinicians: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}") 