import uvicorn
import os
import dotenv
import sys

# Load the environment variables to os.environ
dotenv.load_dotenv()

def main():
    uvicorn.run("api.server:app", host="0.0.0.0", port=int(
        os.environ.get("PORT", 5000)), reload=os.environ.get("RELOAD", False))


if __name__ == "__main__":
    sys.exit(main())
