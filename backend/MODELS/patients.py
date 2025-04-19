from pydantic import BaseModel
from typing import Optional
import uuid

class PatientCreateSchema(BaseModel):
    PatientID: uuid.UUID
    FirstName: str
    LastName: str
    DateOfBirth: str  # or date
    Gender: str
    Email: str