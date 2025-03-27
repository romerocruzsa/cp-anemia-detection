import asyncpg
from backend.models.screening_report import ScreeningReport
from backend.db.database import db_pool
import uuid

class ScreeningReportDAO:
    async def get_by_id(self, screening_report_id: uuid.UUID):
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM screening_reports WHERE id = $1", screening_report_id)
            if row:
                return ScreeningReport(**dict(row))
            return None

    async def create(self, screening_report: ScreeningReport):
        async with db_pool.acquire() as conn:
            screening_report.id = uuid.uuid4()
            await conn.execute(
                """
                INSERT INTO screening_reports (id, visit_id, clinician_observation, physical_findings, impression, recommendations, created_at, approved_by, approved_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7, $8)
                """,
                screening_report.id, screening_report.visit_id, screening_report.clinician_observation,
                screening_report.physical_findings, screening_report.impression, screening_report.recommendations,
                screening_report.approved_by, screening_report.approved_at
            )
            return screening_report

    async def update(self, screening_report_id: uuid.UUID, screening_report: ScreeningReport):
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE screening_reports SET visit_id = $1, clinician_observation = $2, physical_findings = $3, impression = $4, recommendations = $5, approved_by = $6, approved_at = $7
                WHERE id = $8
                """,
                screening_report.visit_id, screening_report.clinician_observation, screening_report.physical_findings,
                screening_report.impression, screening_report.recommendations, screening_report.approved_by,
                screening_report.approved_at, screening_report_id
            )
            return await self.get_by_id(screening_report_id)

    async def delete(self, screening_report_id: uuid.UUID):
        async with db_pool.acquire() as conn:
            await conn.execute("DELETE FROM screening_reports WHERE id = $1", screening_report_id)