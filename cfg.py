import os

user = "dbUser"
password = "dbUserPassword"
DB_URI = f"mongodb+srv://{user}:{password}@best-attributions.oglb5.mongodb.net/best" \
         "-attributions?retryWrites=true&w=majority"
DB_NAME = "best-attributions"

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data")
