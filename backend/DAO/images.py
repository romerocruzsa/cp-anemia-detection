from CONFIG.db_config import pg_config
import asyncpg
import json
import numpy as np

class ImagesDAO: 
    def __init__(self):
        self.db_config = pg_config
        self.pool = None  # Asyncpg connection pool

    async def connect(self):
        """Initialize connection pool if not already created."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.db_config['host'],
                database=self.db_config['dbname'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                port=self.db_config['port']
            )

    async def insert_image(self, image_data):
        """
        Insert a single image record asynchronously into the database.
        """
        if self.pool is None:
            await self.connect()  # Ensure connection pool is available
        
        try:
            async with self.pool.acquire() as conn:
                image_vector = np.array(image_data['IMAGE_VECTOR'], dtype=float).tolist()

                query = """
                INSERT INTO anemia_data(
                    IMAGE_ID, HB_LEVEL, SEVERITY, AGE_MONTHS, GENDER, REMARK, HOSPITAL, 
                    CITY_TOWN, MUNICIPALITY_DISTRICT, REGION, COUNTRY, SEVERITY_CLASS, 
                    IMAGE_PATH, IMAGE_VECTOR
                ) 
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) 
                RETURNING I_ID;
                """
                
                I_ID = await conn.fetchval(query, 
                    image_data['IMAGE_ID'],
                    image_data['HB_LEVEL'],
                    image_data['SEVERITY'],
                    image_data['AGE_MONTHS'],
                    image_data['GENDER'],
                    image_data['REMARK'],
                    image_data['HOSPITAL'],
                    image_data['CITY_TOWN'],
                    image_data['MUNICIPALITY_DISTRICT'],
                    image_data['REGION'],
                    image_data['COUNTRY'],
                    image_data['SEVERITY_CLASS'],
                    image_data['IMAGE_PATH'],
                    image_vector
                )
                return I_ID
        except Exception as e:
            print(f"Error during insert operation: {e}")
            return None

    async def insert_images_from_json(self, json_file_path):
        """
        Insert multiple image records from a JSON file into the database asynchronously.
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            inserted_ids = []
            for entry in data:
                image_id = await self.insert_image(entry)
                if image_id:
                    inserted_ids.append(image_id)
            
            return {"message": "Data added successfully!", "inserted_ids": inserted_ids}
        except Exception as e:
            print("Error loading or inserting data:", e)
            return {"message": "Error occurred during insertion."}

    async def get_images(self):
        """Retrieve all images from the database asynchronously."""
        if self.pool is None:
            await self.connect()
        
        try:
            async with self.pool.acquire() as conn:
                query = "SELECT * FROM anemia_data"
                images = await conn.fetch(query)
                return [dict(image) for image in images]  # Convert to list of dicts
        except Exception as e:
            print("Error during select operation:", e)
            return None

    async def close_connection(self):
        """Close the connection pool."""
        if self.pool:
            await self.pool.close()
