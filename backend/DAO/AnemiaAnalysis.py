from CONFIG.db_config import pg_config
import asyncpg
import logging

logging.basicConfig(level=logging.ERROR)

class AnemiaAnalysisDAO:
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
                logging.error(f"An error occurred while creating the connection pool: {e}")
                raise

    async def getAnalysis(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT AnalysisID, ImageID, AnemiaStatus, ConfidenceScore, AnalysisDate 
                FROM AnemiaAnalysis;
                """
                return await conn.fetch(query)
            except Exception as e:
                logging.error(f"Error fetching uploads: {e}")
                return None

    async def getAnalysisByID(self, image_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT AnalysisID, ImageID, AnemiaStatus, ConfidenceScore, AnalysisDate 
                FROM AnemiaAnalysis WHERE ImageID = $1;
                """
                return await conn.fetchrow(query, image_id)
            except Exception as e:
                logging.error(f"Error fetching analysis for image {image_id}: {e}")
                return None

    async def insertAnalysis(self, image_id, status, confidence):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                INSERT INTO AnemiaAnalysis (ImageID, AnemiaStatus, ConfidenceScore)
                VALUES ($1, $2, $3) RETURNING AnalysisID;
                """
                return await conn.fetchval(query, image_id, status, confidence)
            except Exception as e:
                logging.error(f"Error inserting analysis: {e}")
                return None

    async def getAnalysisByPatient(self, patient_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
            SELECT 
                ia.AnalysisID AS Sample,
                p.PatientID AS "Patient ID",
                TO_CHAR(ia.AnalysisDate, 'YYYY-MM-DD') AS "Date",
                TO_CHAR(ia.AnalysisDate, 'HH24:MI:SS') AS "Time",
                ia.ConfidenceScore AS "Hemoglobin Level",
                ia.AnemiaStatus AS "Remark",
                'Dr. John Doe' AS "Doctor",
                'General Hospital' AS "Hospital"
            FROM 
                AnemiaAnalysis ia
            JOIN 
                ImageUploads iu ON ia.ImageID = iu.ImageID
            JOIN 
                Patients p ON iu.PatientID = p.PatientID
            WHERE 
                p.PatientID = $1;
            """
                return await conn.fetch(query, patient_id)
            except Exception as e:
                logging.error(f"Error fetching analysis for patient {patient_id}: {e}")
                return None
            
    async def deleteAnalysis(self, analysis_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                DELETE FROM AnemiaAnalysis 
                WHERE AnalysisID = $1;
                """
                return await conn.execute(query, analysis_id)
            except Exception as e:
                logging.error(f"Error deleting analysis {analysis_id}: {e}")
                return None
            

    async def updateAnalysis(self, analysis_id, status, confidence):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                UPDATE AnemiaAnalysis 
                SET AnemiaStatus = $1, ConfidenceScore = $2 
                WHERE AnalysisID = $3;
                """
                return await conn.execute(query, status, confidence, analysis_id)
            except Exception as e:
                logging.error(f"Error updating analysis {analysis_id}: {e}")
                return None
            

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")