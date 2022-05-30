# Server file for the API using FastAPI
# Version: 1.0
# Author: Thomi Aditya Alhakiim

import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import tensorflow as tf
from theia import RetrievalModel
import dotenv

app = FastAPI()
config = dotenv.load_dotenv()
model = RetrievalModel()


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>Theia</title>
        </head>
        <body>
            <h1>Theia</h1>
            <p> This is a recommendation engine for mental health apps.</p>
            <p> For more information, visit <a href="#">theia.ai</a>.</p>
        </body>
    </html>
    """

@app.get("/api/v1/recommend/{user_id}")
def recommend(user_id: str):
    result = model.recommend(str(user_id), "last_saved")
    
    # Change the result of tf Tensor to a list
    result = result.numpy().tolist()[0]

    return {"status": "success", "result": result}

def main():
    uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=True)


if __name__ == "__main__":
    sys.exit(main())
