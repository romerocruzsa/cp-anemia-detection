from CONFIG.db_config import pg_config
import asyncpg
import logging

logging.basicConfig(level=logging.ERROR)

class AdministratorsDAO:
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

    async def create_administrator(self, user_id: int, first_name: str, last_name: str):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO Administrators (UserID, FirstName, LastName)
                    VALUES ($1, $2, $3)
                    RETURNING AdminID;
                """
                return await conn.fetchval(query, user_id, first_name, last_name)
            except Exception as e:
                logging.error(f"Error creating administrator: {e}")
                return None

    async def get_administrator_by_id(self, admin_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT a.*, u.Email, u.Role, u.CreatedAt
                    FROM Administrators a
                    JOIN Users u ON a.UserID = u.UserID
                    WHERE a.AdminID = $1;
                """
                return await conn.fetchrow(query, admin_id)
            except Exception as e:
                logging.error(f"Error getting administrator: {e}")
                return None

    async def get_administrator_by_user_id(self, user_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT a.*, u.Email, u.Role, u.CreatedAt
                    FROM Administrators a
                    JOIN Users u ON a.UserID = u.UserID
                    WHERE a.UserID = $1;
                """
                return await conn.fetchrow(query, user_id)
            except Exception as e:
                logging.error(f"Error getting administrator: {e}")
                return None

    async def get_all_administrators(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT a.*, u.Email, u.Role, u.CreatedAt
                    FROM Administrators a
                    JOIN Users u ON a.UserID = u.UserID;
                """
                return await conn.fetch(query)
            except Exception as e:
                logging.error(f"Error getting all administrators: {e}")
                return None 