from CONFIG.db_config import pg_config
import numpy as np
import asyncpg
import logging


logging.basicConfig(level=logging.ERROR)
print(pg_config)
class PatientsDAO:
    def __init__(self):
        self.db_config = pg_config
        self.pool = None

    async def connect(self):
        if self.pool is None:
            try:
                print("Creating connection pool...")
                self.pool = await asyncpg.create_pool(
                    host=self.db_config['host'],
                    database=self.db_config['dbname'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    port=self.db_config['port']
                )
                print("Connection pool created successfully!")
            except Exception as e:
                logging.error(f"An error occurred while creating the connection pool: {e}")
                raise
    
    async def getPatients(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt FROM patients;"
                result = await conn.fetch(query)
                print(result)
                logging.info(f"Query result: {result}")
                return result
            except Exception as e:
                logging.error(f"Error fetching patients: {e}")
                return None

    async def getPatientsById(self, pid):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt FROM Patients WHERE PatientID = $1;"
                return await conn.fetchrow(query, pid)
            except Exception as e:
                logging.error(f"Error fetching patient by ID {pid}: {e}")
                return None

    async def insertPatients(self, fname, lname, dob, gender, email):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                INSERT INTO Patients ( FirstName, LastName, DateOfBirth, Gender, Email)
                VALUES ($1, $2, $3, $4, $5) RETURNING PatientID;
                """
                return await conn.fetchval(query, fname, lname, dob, gender, email)
            except Exception as e:
                logging.error(f"Error inserting patient: {e}")
                return None

    async def deletePatientsById(self, pid):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = "DELETE FROM Patients WHERE PatientID = $1;"
                result = await conn.execute(query, pid)
                return result
            except Exception as e:
                logging.error(f"Error deleting patient {pid}: {e}")
                return None

    async def putPatientsByID(self,pid,fname, lname, dob, gender, email):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                UPDATE Patients SET FirstName = $1, LastName = $2, DateOfBirth = $3, 
                Gender = $4, Email = $5 WHERE PatientID = $6;
                """
                result = await conn.execute(query, fname, lname, dob, gender, email,pid)
                return result
            except Exception as e:
                logging.error(f"Error updating patient {pid}: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")
