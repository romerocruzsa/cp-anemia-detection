from CONFIG.db_config import pg_config
import asyncpg
import logging
import bcrypt
from datetime import datetime

logging.basicConfig(level=logging.ERROR)

class UsersDAO:
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

    async def create_user(self, email: str, password: str, role: str):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                query = """
                    INSERT INTO Users (Email, PasswordHash, Role)
                    VALUES ($1, $2, $3)
                    RETURNING UserID;
                """
                return await conn.fetchval(query, email, hashed_pw, role)
            except Exception as e:
                logging.error(f"Error creating user: {e}")
                return None

    async def validate_login(self, email: str, password: str):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT UserID, PasswordHash, Role
                    FROM Users
                    WHERE Email = $1 AND IsActive = true;
                """
                row = await conn.fetchrow(query, email)
                if row and bcrypt.checkpw(password.encode('utf-8'), row['passwordhash'].encode('utf-8')):
                    # Update last login
                    await conn.execute(
                        "UPDATE Users SET LastLogin = $1 WHERE UserID = $2",
                        datetime.now(), row['userid']
                    )
                    return row['userid'], row['role']
                return None, None
            except Exception as e:
                logging.error(f"Error validating login: {e}")
                return None, None

    async def get_user_by_id(self, user_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT UserID, Email, Role, IsActive, LastLogin
                    FROM Users
                    WHERE UserID = $1;
                """
                return await conn.fetchrow(query, user_id)
            except Exception as e:
                logging.error(f"Error getting user: {e}")
                return None

    async def update_user_status(self, user_id: int, is_active: bool):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    UPDATE Users
                    SET IsActive = $1
                    WHERE UserID = $2;
                """
                return await conn.execute(query, is_active, user_id)
            except Exception as e:
                logging.error(f"Error updating user status: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}") 