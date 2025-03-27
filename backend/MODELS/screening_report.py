from pydantic import BaseModel
from typing import Optional
import uuid
from datetime import datetime
from typing import Dict, Any

class ScreeningReport(BaseModel):
    id: Optional[uuid.UUID] = None
    visit_id: Optional[uuid.UUID] = None
    clinician_observation: Optional[str] = None
    physical_findings: Optional[Dict[str, Any]] = None
    impression: Optional[str] = None
    recommendations: Optional[str] = None
    created_at: Optional[datetime] = None
    approved_by: Optional[uuid.UUID] = None
    approved_at: Optional[datetime] = None