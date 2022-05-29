# Server file for the API using FastAPI
# Version: 1.0
# Author: Thomi Aditya Alhakiim

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


def main():
    uvicorn.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    sys.exit(main())
