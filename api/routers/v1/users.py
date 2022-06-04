from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import HTMLResponse
from ...models.user import User, UserSchema

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

@router.get("/", response_class=HTMLResponse)
def root():
    return """
    <h1>Users</h1>
    <p> This is the users endpoint.</p>
    """

@router.get("/{user_id}")
async def get_user(user_id: str):
    return {"status": "success", "result": User.get(user_id)}

@router.post("/")
async def create_user(user: UserSchema):
    User.create(dict(user))
    return {"status": "success", "result": user}