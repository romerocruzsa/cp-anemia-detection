from DAO.medical_notes import MedicalNotesDAO
import logging
from datetime import datetime

logging.basicConfig(level=logging.ERROR)

class MedicalNotesHandler:
    def __init__(self):
        self.dao = MedicalNotesDAO()

    async def create_note(self, clinician_id: int, patient_id: int, analysis_id: int,
                         note_text: str, follow_up_date: str = None):
        try:
            note_id = await self.dao.create_note(
                clinician_id, patient_id, analysis_id, note_text, follow_up_date
            )
            return note_id
        except Exception as e:
            logging.error(f"Error creating medical note: {e}")
            return None

    async def get_patient_notes(self, patient_id: int):
        try:
            notes = await self.dao.get_patient_notes(patient_id)
            if notes:
                return [dict(note) for note in notes]
            return []
        except Exception as e:
            logging.error(f"Error getting patient notes: {e}")
            return []

    async def get_note_by_id(self, note_id: int):
        try:
            note = await self.dao.get_note_by_id(note_id)
            return dict(note) if note else None
        except Exception as e:
            logging.error(f"Error getting note by ID: {e}")
            return None

    async def update_note(self, note_id: int, note_text: str, follow_up_date: str = None):
        try:
            updated_id = await self.dao.update_note(note_id, note_text, follow_up_date)
            return updated_id
        except Exception as e:
            logging.error(f"Error updating medical note: {e}")
            return None

    async def get_pending_follow_ups(self, clinician_id: int = None):
        try:
            follow_ups = await self.dao.get_pending_follow_ups(clinician_id)
            if follow_ups:
                return [dict(follow_up) for follow_up in follow_ups]
            return []
        except Exception as e:
            logging.error(f"Error getting pending follow-ups: {e}")
            return []

    async def close(self):
        await self.dao.close_connection() 