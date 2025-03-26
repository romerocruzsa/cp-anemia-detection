from CONFIG.db_config import pg_config
import psycopg2
import json
import numpy as np

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
        query = "SELECT PatientID, FirstName, LastName , DateOfBirth, Gender , Email,CreatedAt FROM Patients;"
        try:
            cursor.execute(query)
            result = []
            for row in cursor:
                result.append(row)
            cursor.close()
            return result
        except Exception as e:
            print("An error occurred: ", e)
        finally:
            cursor.close()

    def getPatientById(self,pid):
        cursor = self.conn.cursor()
        query = "SELECT * FROM patients where e_id =%s"
        try:
            cursor.execute(query, (pid,))
            result = cursor.fetchone()
            return result
        except Exception as e:
            print("An error occurred: ", e)
        finally:
            cursor.close()
    
