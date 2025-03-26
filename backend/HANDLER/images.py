import json
from DAO.images import ImagesDAO  # Import the Images class from the DAO module

class DataHandler:

    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.dao = ImagesDAO()  # Create an instance of the Images class, not the module

    async def load_and_insert_data(self):
        """
        Load data from JSON and insert it into the database.
        """
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)

            inserted_ids = []
            for entry in data:
                image_id = await self.dao.insert_image(entry)
                if image_id:
                    inserted_ids.append(image_id)
            return {"message": "Data inserted successfully!", "inserted_ids": inserted_ids}
        except Exception as e:
            print(f"Error during data processing: {e}")
            return {"message": "Error during data processing"}

    def fetch_data(self):
        """Fetch and print the data from the database."""
        images = self.dao.get_images()
        if images:
            for image in images:
                print(image)
        else:
            print("No data found.")

    def close_connection(self):
        """Close the database connection."""
        self.dao.close_connection()
