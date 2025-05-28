from CONFIG.db_config import pg_config
import asyncpg
import logging

logging.basicConfig(level=logging.ERROR)

class MedicalNotesDAO:
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

    async def create_note(self, clinician_id: int, patient_id: int, analysis_id: int, 
                         note_text: str, follow_up_date: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    INSERT INTO MedicalNotes 
                    (ClinicianID, PatientID, AnalysisID, NoteText, FollowUpDate)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING NoteID;
                """
                return await conn.fetchval(
                    query, clinician_id, patient_id, analysis_id, note_text, follow_up_date
                )
            except Exception as e:
                logging.error(f"Error creating medical note: {e}")
                return None

    async def get_patient_notes(self, patient_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT m.*, c.FirstName as ClinicianFirstName, c.LastName as ClinicianLastName,
                           a.AnemiaStatus, a.ConfidenceScore
                    FROM MedicalNotes m
                    JOIN Clinicians c ON m.ClinicianID = c.ClinicianID
                    JOIN AnemiaAnalysis a ON m.AnalysisID = a.AnalysisID
                    WHERE m.PatientID = $1
                    ORDER BY m.CreatedAt DESC;
                """
                return await conn.fetch(query, patient_id)
            except Exception as e:
                logging.error(f"Error getting patient notes: {e}")
                return None

    async def get_note_by_id(self, note_id: int):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    SELECT m.*, c.FirstName as ClinicianFirstName, c.LastName as ClinicianLastName,
                           a.AnemiaStatus, a.ConfidenceScore
                    FROM MedicalNotes m
                    JOIN Clinicians c ON m.ClinicianID = c.ClinicianID
                    JOIN AnemiaAnalysis a ON m.AnalysisID = a.AnalysisID
                    WHERE m.NoteID = $1;
                """
                return await conn.fetchrow(query, note_id)
            except Exception as e:
                logging.error(f"Error getting note by ID: {e}")
                return None

    async def update_note(self, note_id: int, note_text: str, follow_up_date: str = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                query = """
                    UPDATE MedicalNotes 
                    SET NoteText = $1, FollowUpDate = $2, UpdatedAt = CURRENT_TIMESTAMP
                    WHERE NoteID = $3
                    RETURNING NoteID;
                """
                return await conn.fetchval(query, note_text, follow_up_date, note_id)
            except Exception as e:
                logging.error(f"Error updating medical note: {e}")
                return None

    async def get_pending_follow_ups(self, clinician_id: int = None):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                conditions = ["FollowUpDate <= CURRENT_DATE"]
                params = []
                param_count = 1

                if clinician_id:
                    conditions.append(f"ClinicianID = ${param_count}")
                    params.append(clinician_id)
                    param_count += 1

                where_clause = " AND ".join(conditions)
                query = f"""
                    SELECT m.*, p.FirstName as PatientFirstName, p.LastName as PatientLastName,
                           c.FirstName as ClinicianFirstName, c.LastName as ClinicianLastName
                    FROM MedicalNotes m
                    JOIN Patients p ON m.PatientID = p.PatientID
                    JOIN Clinicians c ON m.ClinicianID = c.ClinicianID
                    WHERE {where_clause}
                    ORDER BY m.FollowUpDate ASC;
                """
                return await conn.fetch(query, *params)
            except Exception as e:
                logging.error(f"Error getting pending follow-ups: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}") 