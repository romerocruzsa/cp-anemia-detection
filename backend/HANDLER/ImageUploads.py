from fastapi.responses import JSONResponse
from DAO.ImageUploads import ImageUploadsDAO

class ImageUploadsHandler:
    def mapToDict(self, t):
        return {
            'ImageID': t["imageid"],
            'PatientID': t["patientid"],
            'ImagePath': t["imagepath"],
            'UploadDate': t["uploaddate"].isoformat() if t["uploaddate"] else None,
            'Status': t["status"]
        }
    
    async def getAllUploads(self):
        dao = ImageUploadsDAO()
        try:
            dbtuples = await dao.getUploads()
            result = [self.mapToDict(e) for e in dbtuples] if dbtuples else []
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(content={'error': f'An error occurred while retrieving Uploads: {e}'}, status_code=500)

    async def getUploadsByPatient(self, patient_id):
        dao = ImageUploadsDAO()
        try:
            uploads = await dao.getUploadsByPatient(patient_id)
            if uploads:
                result = [self.mapToDict(u) for u in uploads]
                return JSONResponse(content=result)
            return JSONResponse(content=[], status_code=200)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error retrieving uploads: {e}'}, 
                status_code=500
            )

    async def createUpload(self, patient_id: int, image_path: str):
        dao = ImageUploadsDAO()
        try:
            image_id = await dao.insertUpload(patient_id, image_path)
            if image_id:
                return JSONResponse(content={
                    "ImageID": image_id,
                    "PatientID": patient_id,
                    "ImagePath": image_path
                })
            return JSONResponse(
                content={"error": "Failed to create upload"}, 
                status_code=500
            )
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error creating upload: {e}'}, 
                status_code=500
            )

    async def updateUploadStatus(self, image_id: int, status: str):
        dao = ImageUploadsDAO()
        try:
            result = await dao.updateStatus(image_id, status)
            if result:
                return JSONResponse(content={"message": "Status updated successfully"})
            return JSONResponse(content={"error": "Upload not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error updating status: {e}'}, 
                status_code=500
            )
        
    async def updateImage(self, image_id: int, image_path: str):
        dao = ImageUploadsDAO()
        try:
            result = await dao.updateImage(image_id, image_path)
            if result:
                return JSONResponse(content={"message": "Image updated successfully"})
            return JSONResponse(content={"error": "Image not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error updating image: {e}'},
                status_code=500
            )

    async def deleteImage(self, image_id: int):
        dao = ImageUploadsDAO()
        try:
            result = await dao.deleteImage(image_id)
            if result:
                return JSONResponse(content={"message": "Image deleted successfully"})
            return JSONResponse(content={"error": "Image not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error deleting image: {e}'},
                status_code=500
            )