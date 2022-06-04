# Server file for the API using FastAPI
# Version: 1.0
# Author: Thomi Aditya Alhakiim

import os
import sys
import time
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import dotenv
from .routers import home, v1

app = FastAPI()

# Load the environment variables to os.environ
dotenv.load_dotenv()

app.include_router(home.router)
app.include_router(v1.router)