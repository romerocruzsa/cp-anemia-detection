import asyncpg
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv(os.path.join("CONFIG", "local.env"))

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL must be set in CONFIG/local.env")

# Create a connection pool
pool = None

async def get_pool():
    """
    Get or create the database connection pool.
    """
    global pool
    if pool is None:
        try:
            pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=1,
                max_size=10
            )
        except Exception as e:
            logging.error(f"Error creating database pool: {e}")
            raise
    return pool

async def get_connection():
    """
    Get a database connection from the pool.
    """
    try:
        pool = await get_pool()
        return await pool.acquire()
    except Exception as e:
        logging.error(f"Error getting database connection: {e}")
        raise

async def release_connection(conn):
    """
    Release a database connection back to the pool.
    """
    try:
        if pool is not None:
            await pool.release(conn)
    except Exception as e:
        logging.error(f"Error releasing database connection: {e}")
        raise

async def close_pool():
    """
    Close all connections in the pool.
    """
    global pool
    if pool is not None:
        await pool.close()
        pool = None 