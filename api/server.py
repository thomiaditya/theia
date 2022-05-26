# Server file for the API using FastAPI
# Version: 1.0
# Author: Thomi Aditya Alhakiim

import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


if __name__ == "app":
    uvicorn.run("api.server:app", host="127.0.0.1",
                port=5000, log_level="info")
