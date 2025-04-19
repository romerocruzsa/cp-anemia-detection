from CONFIG.db_config import pg_config
import psycopg2
from encryption import decrypt_field, encrypt_field

class PatientsDAO:

    def __init__(self):
        try:
            self.conn = psycopg2.connect(
                host=pg_config['host'],
                database=pg_config['dbname'],
                user=pg_config['user'],
                password=pg_config['password'],
                port=pg_config['port']
            )
        except Exception as e:
            print("Error while connecting to PostgreSQL", e)
            self.conn = None
    
    def getPatients(self):
        cursor = self.conn.cursor()
        query = "SELECT PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt FROM Patients;"
        try:
            cursor.execute(query)
            result = []
            for row in cursor:
                decrypted_row = list(row)
                # Decrypt PII fields by index
                decrypted_row[1] = decrypt_field(decrypted_row[1])  # FirstName
                decrypted_row[2] = decrypt_field(decrypted_row[2])  # LastName
                decrypted_row[3] = decrypt_field(decrypted_row[3])  # DateOfBirth
                decrypted_row[5] = decrypt_field(decrypted_row[5])  # Email
                result.append(tuple(decrypted_row))
            cursor.close()
            return result
        except Exception as e:
            print("An error occurred: ", e)
        finally:
            cursor.close()

    def getPatientById(self, pid):
        cursor = self.conn.cursor()
        query = "SELECT * FROM patients WHERE e_id = %s"
        try:
            cursor.execute(query, (pid,))
            row = cursor.fetchone()
            if row:
                decrypted_row = list(row)
                decrypted_row[2] = decrypt_field(decrypted_row[2])  # FirstName
                decrypted_row[3] = decrypt_field(decrypted_row[3])  # LastName
                decrypted_row[4] = decrypt_field(decrypted_row[4])  # DateOfBirth
                decrypted_row[6] = decrypt_field(decrypted_row[6])  # Email
                return tuple(decrypted_row)
            return None
        except Exception as e:
            print("An error occurred: ", e)
        finally:
            cursor.close()

    def createPatient(self, patient_data: dict):
        """
        Inserts a new patient record with encryption for PII fields.
        Expected keys: PatientID, FirstName, LastName, DateOfBirth, Gender, Email
        """
        cursor = self.conn.cursor()
        try:
            query = """
                INSERT INTO Patients (
                    PatientID, FirstName, LastName, DateOfBirth, Gender, Email, CreatedAt
                ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(query, (
                patient_data['PatientID'],
                encrypt_field(patient_data['FirstName']),
                encrypt_field(patient_data['LastName']),
                encrypt_field(patient_data['DateOfBirth']),
                patient_data['Gender'],
                encrypt_field(patient_data['Email']),
            ))
            self.conn.commit()
            return {"message": "Patient inserted successfully."}
        except Exception as e:
            print("Error inserting patient:", e)
            self.conn.rollback()
            return {"error": "Failed to insert patient."}
        finally:
            cursor.close()
