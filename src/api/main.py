import os
from fastapi import FastAPI
from google.cloud import storage
from google.oauth2 import service_account
from .routes import router

# Get environment variables
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')

if not credentials_path or not project_id:
    raise ValueError(
        f"Missing required environment variables:\n"
        f"GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}\n"
        f"GOOGLE_CLOUD_PROJECT: {project_id}"
    )

# Initialize credentials properly
credentials = service_account.Credentials.from_service_account_file(
    credentials_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize Google Cloud client with explicit credentials
storage_client = storage.Client(
    project=project_id,
    credentials=credentials
)

app = FastAPI(
    title="Sheetz Recommendation API",
    description="API for personalized product recommendations",
    version="1.0.0"
)

app.include_router(
    router,
    prefix="/api/v1",
    tags=["recommendations"]
)