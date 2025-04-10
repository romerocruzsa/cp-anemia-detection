from CONFIG.db_config import pg_config
import asyncpg
import logging

logging.basicConfig(level=logging.ERROR)

class ImageUploadsDAO:
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

    async def getUploads(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT ImageID, PatientID, ImagePath, UploadDate, Status 
                FROM ImageUploads;
                """
                return await conn.fetch(query)
            except Exception as e:
                logging.error(f"Error fetching uploads: {e}")
                return None

    async def getUploadById(self, image_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT ImageID, PatientID, ImagePath, UploadDate, Status 
                FROM ImageUploads WHERE ImageID = $1;
                """
                return await conn.fetchrow(query, image_id)
            except Exception as e:
                logging.error(f"Error fetching upload by ID {image_id}: {e}")
                return None

    async def getUploadsByPatient(self, patient_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                SELECT ImageID, PatientID, ImagePath, UploadDate, Status 
                FROM ImageUploads WHERE PatientID = $1;
                """
                return await conn.fetch(query, patient_id)
            except Exception as e:
                logging.error(f"Error fetching uploads for patient {patient_id}: {e}")
                return None

    async def insertUpload(self, patient_id, image_path):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                INSERT INTO ImageUploads (PatientID, ImagePath)
                VALUES ($1, $2) RETURNING ImageID;
                """
                return await conn.fetchval(query, patient_id, image_path)
            except Exception as e:
                logging.error(f"Error inserting upload: {e}")
                return None

    async def updateStatus(self, image_id, status):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                UPDATE ImageUploads SET Status = $1 
                WHERE ImageID = $2;
                """
                return await conn.execute(query, status, image_id)
            except Exception as e:
                logging.error(f"Error updating status for image {image_id}: {e}")
                return None
            

    async def updateImage(self, image_id, image_path):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                UPDATE ImageUploads 
                SET ImagePath = $1 
                WHERE ImageID = $2;
                """
                return await conn.execute(query, image_path, image_id)
            except Exception as e:
                logging.error(f"Error updating image {image_id}: {e}")
                return None
            
    async def deleteImage(self, image_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                DELETE FROM ImageUploads 
                WHERE ImageID = $1;
                """
                return await conn.execute(query, image_id)
            except Exception as e:
                logging.error(f"Error deleting image {image_id}: {e}")
                return None