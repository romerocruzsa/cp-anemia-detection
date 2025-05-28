from CONFIG.db_config import pg_config
import asyncpg
import logging
import os
import bcrypt
from encryption import AESCipher
from datetime import datetime

logging.basicConfig(level=logging.ERROR)

class PatientsDAO:
    def __init__(self):
        self.db_config = pg_config
        self.pool = None

        # ── Load and validate a 32-byte AES key from env ──
        aes_key = os.getenv("AES_KEY")
        if not aes_key:
            raise ValueError("AES_KEY environment variable is required")
        key_bytes = aes_key.encode('utf-8')
        if len(key_bytes) != 32:
            raise ValueError("AES_KEY must be exactly 32 bytes for AES-256")
        self.cipher = AESCipher(key_bytes)

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

    async def getPatients(self):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                sql = """
                  SELECT PatientID, FirstName, LastName, DateOfBirth,
                         Gender, CreatedAt
                    FROM Patients;
                """
                rows = await conn.fetch(sql)
                decrypted = []
                for r in rows:
                    rec = dict(r)
                    # decrypt first name and last name
                    rec['firstname'] = self.cipher.decrypt(rec['firstname'])
                    rec['lastname']  = self.cipher.decrypt(rec['lastname'])
                    decrypted.append(rec)
                return decrypted
            except Exception as e:
                logging.error(f"Error decrypting patients: {e}")
                return None

    async def getPatientsById(self, pid):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                sql = """
                  SELECT PatientID, FirstName, LastName, DateOfBirth,
                         Gender, CreatedAt
                    FROM Patients
                   WHERE PatientID = $1;
                """
                row = await conn.fetchrow(sql, pid)
                if not row:
                    return None
                rec = dict(row)
                # decrypt first name and last name
                rec['firstname'] = self.cipher.decrypt(rec['firstname'])
                rec['lastname']  = self.cipher.decrypt(rec['lastname'])
                return rec
            except Exception as e:
                logging.error(f"Error fetching/decrypting patient {pid}: {e}")
                return None

    async def insertPatientWithPassword(self, user_id, fname, lname, dob, gender, email, password):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                # hash the password
                hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                # encrypt first name and last name
                encrypted_fname = self.cipher.encrypt(fname)
                encrypted_lname = self.cipher.encrypt(lname)

                # Convert date string to date object
                dob_date = datetime.strptime(dob, '%Y-%m-%d').date()

                sql = """
                  INSERT INTO Patients
                    (UserID, FirstName, LastName, DateOfBirth, Gender)
                  VALUES ($1, $2, $3, $4, $5)
                  RETURNING PatientID;
                """
                return await conn.fetchval(sql,
                    user_id,
                    encrypted_fname,
                    encrypted_lname,
                    dob_date,
                    gender
                )
            except Exception as e:
                logging.error(f"Error inserting patient with encrypted fields: {e}")
                return None

    async def validatePatientLogin(self, email, password):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                sql = "SELECT PatientID, Email, Password FROM Patients;"
                rows = await conn.fetch(sql)
                for r in rows:
                    raw = r['email']
                    # Try to decrypt; if it fails, skip this row
                    try:
                        dec_email = self.cipher.decrypt(raw)
                    except Exception:
                        logging.warning(f"Skipping non‐ciphertext email for patient {r['patientid']}")
                        continue

                    # If the decrypted email matches, check the password
                    if dec_email == email and bcrypt.checkpw(
                        password.encode('utf-8'),
                        r['password'].encode('utf-8')
                    ):
                        return r['patientid']
                return None
            except Exception as e:
                logging.error(f"Login error for {email}: {e}")
                return None

    async def putPatientsByID(self, pid, fname, lname, dob, gender, email):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                # encrypt first/last names and email before update
                encrypted_fname = self.cipher.encrypt(fname)
                encrypted_lname = self.cipher.encrypt(lname)
                encrypted_email = self.cipher.encrypt(email)

                sql = """
                  UPDATE Patients
                     SET FirstName   = $1,
                         LastName    = $2,
                         DateOfBirth = $3,
                         Gender      = $4,
                         Email       = $5
                   WHERE PatientID = $6;
                """
                return await conn.execute(sql,
                    encrypted_fname,
                    encrypted_lname,
                    dob,
                    gender,
                    encrypted_email,
                    pid
                )
            except Exception as e:
                logging.error(f"Error updating patient {pid}: {e}")
                return None

    async def deletePatientsById(self, pid):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                sql = "DELETE FROM Patients WHERE PatientID = $1;"
                return await conn.execute(sql, pid)
            except Exception as e:
                logging.error(f"Error deleting patient {pid}: {e}")
                return None

    async def getPatientsByUserId(self, user_id):
        await self.connect()
        async with self.pool.acquire() as conn:
            try:
                sql = """
                  SELECT PatientID, FirstName, LastName, DateOfBirth,
                         Gender, CreatedAt
                    FROM Patients
                   WHERE UserID = $1;
                """
                row = await conn.fetchrow(sql, user_id)
                if not row:
                    return None
                rec = dict(row)
                # decrypt first name and last name
                rec['firstname'] = self.cipher.decrypt(rec['firstname'])
                rec['lastname']  = self.cipher.decrypt(rec['lastname'])
                return rec
            except Exception as e:
                logging.error(f"Error fetching/decrypting patient for user {user_id}: {e}")
                return None

    async def close_connection(self):
        if self.pool:
            try:
                await self.pool.close()
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")
