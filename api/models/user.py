from datetime import datetime
from enum import Enum
import pydantic
from ..config.database import db

users = db.users

class Gender(str, Enum):
    male = 'M'
    female = 'F'

# User Schema
class UserSchema(pydantic.BaseModel):
    id: int
    name: str
    email: str
    phone: str = "000"
    age: int
    gender: Gender
    created_at: datetime = datetime.today()
    updated_at: datetime = datetime.today()
    deleted_at: datetime = datetime.today()

# The model
class User:
    def create(user):
        users.insert_one(user)