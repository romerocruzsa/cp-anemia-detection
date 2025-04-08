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

    async def getAnalysis(self, image_id):
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
                SELECT a.AnalysisID, a.ImageID, a.AnemiaStatus, 
                       a.ConfidenceScore, a.AnalysisDate, i.ImagePath
                FROM AnemiaAnalysis a
                JOIN ImageUploads i ON a.ImageID = i.ImageID
                WHERE i.PatientID = $1;
                """
                return await conn.fetch(query, patient_id)
            except Exception as e:
                logging.error(f"Error fetching analysis for patient {patient_id}: {e}")
                return None