import pymongo as pm
import dotenv
import os

# Load the environment variables to os.environ
dotenv.load_dotenv()

# Create a connection to the database
client = pm.MongoClient(os.environ.get("DB_STRING"))
db = client['mental-app']

print("Connected to the database")
print("Database name: {}".format(db.name))