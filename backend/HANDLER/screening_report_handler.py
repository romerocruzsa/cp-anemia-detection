from backend.daos.screening_report_dao import ScreeningReportDAO
from backend.models.screening_report import ScreeningReport
import uuid

class ScreeningReportHandler:
    def __init__(self):
        self.screening_report_dao = ScreeningReportDAO()

    async def get_screening_report_by_id(self, screening_report_id: str):
        try:
            screening_report_id_uuid = uuid.UUID(screening_report_id)
        except ValueError:
            return None
        return await self.screening_report_dao.get_by_id(screening_report_id_uuid)

    async def create_screening_report(self, screening_report: ScreeningReport):
        return await self.screening_report_dao.create(screening_report)

    async def update_screening_report(self, screening_report_id: str, screening_report: ScreeningReport):
        try:
            screening_report_id_uuid = uuid.UUID(screening_report_id)
        except ValueError:
            return None
        return await self.screening_report_dao.update(screening_report_id_uuid, screening_report)

    async def delete_screening_report(self, screening_report_id: str):
        try:
            screening_report_id_uuid = uuid.UUID(screening_report_id)
        except ValueError:
            return False
        return await self.screening_report_dao.delete(screening_report_id_uuid)