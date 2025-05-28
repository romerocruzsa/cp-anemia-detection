from db.database import get_connection, release_connection
import bcrypt
from datetime import datetime
import logging

logging.basicConfig(level=logging.ERROR)

class UsersDAO:
    async def create_user(self, email: str, password: str, role: str) -> int:
        try:
            # Hash password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

            # Insert into Users table
            query = """
                INSERT INTO Users (Email, PasswordHash, Role, CreatedAt)
                VALUES ($1, $2, $3, $4)
                RETURNING UserID
            """
            conn = await get_connection()
            try:
                user_id = await conn.fetchval(query, email, hashed_password.decode('utf-8'), role, datetime.now())
                return user_id
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error creating user: {e}")
            return None

    async def validate_login(self, email: str, password: str) -> tuple:
        try:
            query = """
                SELECT UserID, PasswordHash, Role
                FROM Users
                WHERE Email = $1
            """
            conn = await get_connection()
            try:
                result = await conn.fetchrow(query, email)
                
                if not result:
                    return None, None

                user_id, stored_hash, role = result
                
                # Verify password
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                    return user_id, role
                return None, None
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error validating login: {e}")
            return None, None

    async def get_user_by_id(self, user_id: int) -> dict:
        try:
            query = """
                SELECT UserID, Email, Role, CreatedAt
                FROM Users
                WHERE UserID = $1
            """
            conn = await get_connection()
            try:
                result = await conn.fetchrow(query, user_id)
                
                if not result:
                    return None

                return {
                    'UserID': result['userid'],
                    'Email': result['email'],
                    'Role': result['role'],
                    'CreatedAt': result['createdat']
                }
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error getting user: {e}")
            return None

    async def update_user_password(self, user_id: int, new_password: str) -> bool:
        try:
            # Hash new password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)

            query = """
                UPDATE Users
                SET PasswordHash = $1
                WHERE UserID = $2
            """
            conn = await get_connection()
            try:
                await conn.execute(query, hashed_password.decode('utf-8'), user_id)
                return True
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error updating password: {e}")
            return False

    async def delete_user(self, user_id: int) -> bool:
        try:
            query = """
                DELETE FROM Users
                WHERE UserID = $1
            """
            conn = await get_connection()
            try:
                await conn.execute(query, user_id)
                return True
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error deleting user: {e}")
            return False

    async def update_user_status(self, user_id: int, is_active: bool):
        try:
            query = """
                UPDATE Users
                SET IsActive = $1
                WHERE UserID = $2
            """
            conn = await get_connection()
            try:
                await conn.execute(query, is_active, user_id)
                return True
            finally:
                await release_connection(conn)

        except Exception as e:
            logging.error(f"Error updating user status: {e}")
            return False

    async def close_connection(self):
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}") 