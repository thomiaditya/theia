from datetime import datetime
from enum import Enum
import pydantic
from ..config.database import db

therapist = db.therapists

class TherapistSchema(pydantic.BaseModel):
    id: int
    name: str
    email: str
    phone: str = "000"
    age: int