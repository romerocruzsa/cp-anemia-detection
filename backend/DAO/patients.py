from CONFIG.db_config import pg_config
import json
import asyncpg
import numpy as np

class PatientsDAO:

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
                print.error(f"An error occurred: {e}")
    
    async def getPatients(self):
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt FROM Patients;"
                return await conn.fetch(query)
            except Exception as e:
                print.error(f"Error fetching patients: {e}")
                return None

    async def getPatientsById(self, pid):
        async with self.pool.acquire() as conn:
            try:
                query = "SELECT PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt FROM Patients WHERE PatientID = $1;"
                return await conn.fetchrow(query, pid)
            except Exception as e:
                print.error(f"Error fetching patient by ID {pid}: {e}")
                return None

    async def insertPatients(self, id, fname, lname, dob, gender, email, createdAt):
        async with self.pool.acquire() as conn:
            try:
                query = """
                INSERT INTO Patients (PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt)
                VALUES ($1, $2, $3, $4, $5, $6, $7) RETURNING PatientID;
                """
                return await conn.fetchval(query, id, fname, lname, dob, gender, email, createdAt)
            except Exception as e:
                print.error(f"Error inserting patient: {e}")
                return None

    async def deletePatientsById(self, pid):
        async with self.pool.acquire() as conn:
            try:
                query = "DELETE FROM Patients WHERE PatientID = $1;"
                result = await conn.execute(query, pid)
                return result
            except Exception as e:
                print.error(f"Error deleting patient {pid}: {e}")
                return None

    async def putPatientsByID(self, pid, fname, lname, dob, gender, email, createdAt):
        async with self.pool.acquire() as conn:
            try:
                query = """
                UPDATE Patients SET FirstName = $1, LastName = $2, DateOfBirth = $3, 
                Gender = $4, Email = $5, CreatedAt = $6 WHERE PatientID = $7;
                """
                result = await conn.execute(query, fname, lname, dob, gender, email, createdAt, pid)
                return result
            except Exception as e:
                print.error(f"Error updating patient {pid}: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                print.error(f"Error closing database connection: {e}")
    
