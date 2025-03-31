import asyncpg
from backend.config import DATABASE_URL

async def create_db_pool():
    try:
        pool = await asyncpg.create_pool(DATABASE_URL)
        print("Database connection pool created")
        return pool
    except Exception as e:
        print(f"Error creating database pool: {e}")
        return None

db_pool = None  # Initialize the database pool