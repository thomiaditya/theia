from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from ...middleware import authenticate
from ...models.user import User, UserSchema
import csv

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


def read_and_create_users(file: UploadFile = File(...)):
    import os
    from alive_progress import alive_bar

    # Print the file name
    print("File name: {}".format(file.filename))

    # Write the file to a temporary file
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    with alive_bar(len(file.file.readlines())) as bar:
        # Read the file and create users
        with open(file.filename, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["age"] = int(row["age"])
                User.create(dict(UserSchema(**row)))
                bar()

    # Delete the file
    os.remove(file.filename)


@router.post("/csv")
async def create_users_csv(csv_file: UploadFile, pin: str, background_tasks: BackgroundTasks):
    authenticate(pin)
    background_tasks.add_task(read_and_create_users, csv_file)
    return {"status": "success"}

# TODO: Adding more routes for users
