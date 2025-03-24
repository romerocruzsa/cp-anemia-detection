from CONFIG.db_config import pg_config
import psycopg2
import json

class Images: 

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

    def insert_image(self, image_data):
        """
        Insert a single image record into the database.
        """
        if self.conn is None:
            return "Database connection failed"
        
        try:
            cursor = self.conn.cursor()
            query = """
            INSERT INTO anemia_data(
                IMAGE_ID, HB_LEVEL, Severity, Age_Months, GENDER, REMARK, HOSPITAL, 
                CITY_TOWN, MUNICIPALITY_DISTRICT, REGION, COUNTRY, SEVERITY_CLASS, 
                IMAGE_PATH, IMAGE_VECTOR
            ) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            RETURNING I_ID;
            """
            values = (
                image_data['IMAGE_ID'],
                image_data['HB_LEVEL'],
                image_data['Severity'],
                image_data['Age(Months)'],
                image_data['GENDER'],
                image_data['REMARK'],
                image_data['HOSPITAL'],
                image_data['CITY/TOWN'],
                image_data['MUNICIPALITY/DISTRICT'],
                image_data['REGION'],
                image_data['COUNTRY'],
                image_data['SEVERITY_CLASS'],
                image_data['IMAGE_PATH'],
                image_data['IMAGE_VECTOR']
            )
            
            cursor.execute(query, values)
            I_ID = cursor.fetchone()[0]  # Get the inserted I_ID
            self.conn.commit()
            cursor.close()
            return I_ID
        except Exception as e:
            print("Error during insert operation:", e)
            self.conn.rollback()
            return None

    def insert_images_from_json(self, json_file_path):
        """
        Insert multiple image records from a JSON file into the database.
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            inserted_ids = []
            for entry in data:
                image_id = self.insert_image(entry)
                if image_id:
                    inserted_ids.append(image_id)
            
            return {"message": "Data added successfully!", "inserted_ids": inserted_ids}
        except Exception as e:
            print("Error loading or inserting data:", e)
            return {"message": "Error occurred during insertion."}

    def get_images(self):
        """Retrieve all images from the database."""
        if self.conn is None:
            return "Database connection failed"
        try:
            cursor = self.conn.cursor()
            query = "SELECT * FROM incoming_images"
            cursor.execute(query)
            images = cursor.fetchall()
            cursor.close()
            return images
        except Exception as e:
            print("Error during select operation:", e)
            return None

    def close_connection(self):
        """Ensure the connection is properly closed."""
        if self.conn:
            self.conn.close()