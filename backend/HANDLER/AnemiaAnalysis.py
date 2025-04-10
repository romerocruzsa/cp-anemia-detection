from fastapi.responses import JSONResponse
from DAO.AnemiaAnalysis import AnemiaAnalysisDAO

class AnemiaAnalysisHandler:
    def mapToDict(self, t):
        return {
            'AnalysisID': t["analysisid"],
            'ImageID': t["imageid"],
            'AnemiaStatus': t["anemiastatus"],
            'ConfidenceScore': float(t["confidencescore"]),
            'AnalysisDate': t["analysisdate"].isoformat() if t["analysisdate"] else None,
            'ImagePath': t.get("imagepath")
        }
    
    async def getAnalysis(self):
        dao = AnemiaAnalysisDAO()
        try:
            dbtuples = await dao.getAnalysis()
            result = [self.mapToDict(e) for e in dbtuples] if dbtuples else []
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(content={'error': f'An error occurred while retrieving Uploads: {e}'}, status_code=500)

    async def getAnalysisByImage(self, image_id):
        dao = AnemiaAnalysisDAO()
        try:
            analysis = await dao.getAnalysisByID(image_id)
            if analysis:
                return JSONResponse(content=self.mapToDict(analysis))
            return JSONResponse(content={"error": "Analysis not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error retrieving analysis: {e}'}, 
                status_code=500
            )

    async def createAnalysis(self, image_id: int, status: str, confidence: float):
        dao = AnemiaAnalysisDAO()
        try:
            if not (0 <= confidence <= 1):
                return JSONResponse(
                    content={"error": "Confidence score must be between 0 and 1"}, 
                    status_code=400
                )
            
            analysis_id = await dao.insertAnalysis(image_id, status, confidence)
            if analysis_id:
                return JSONResponse(content={
                    "AnalysisID": analysis_id,
                    "ImageID": image_id,
                    "AnemiaStatus": status,
                    "ConfidenceScore": confidence
                })
            return JSONResponse(
                content={"error": "Failed to create analysis"}, 
                status_code=500
            )
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error creating analysis: {e}'}, 
                status_code=500
            )

    async def getPatientHistory(self, patient_id):
        dao = AnemiaAnalysisDAO()
        try:
            analyses = await dao.getAnalysisByPatient(patient_id)
            if analyses:
                result = [self.mapToDict(a) for a in analyses]
                return JSONResponse(content=result)
            return JSONResponse(content=[], status_code=200)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error retrieving patient history: {e}'}, 
                status_code=500
            )

    async def deleteAnalysis(self, analysis_id: int):
        dao = AnemiaAnalysisDAO()
        try:
            result = await dao.deleteAnalysis(analysis_id)
            if result:
                return JSONResponse(content={"message": "Analysis deleted successfully"})
            return JSONResponse(content={"error": "Analysis not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error deleting analysis: {e}'},
                status_code=500
            )

    async def updateAnalysis(self, analysis_id: int, status: str, confidence: float):
        dao = AnemiaAnalysisDAO()
        try:
            result = await dao.updateAnalysis(analysis_id, status, confidence)
            if result:
                return JSONResponse(content={"message": "Analysis updated successfully"})
            return JSONResponse(content={"error": "Analysis not found"}, status_code=404)
        except Exception as e:
            return JSONResponse(
                content={'error': f'Error updating analysis: {e}'},
                status_code=500
            )


