from DB.database import db_pool
from datetime import datetime

async def log_action(user_email: str, action: str, resource: str, resource_id: str = None):
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_log (user_email, action, resource, resource_id)
                VALUES ($1, $2, $3, $4)
                """,
                user_email, action, resource, resource_id
            )
    except Exception as e:
        print(f"Failed to log audit action: {e}")
