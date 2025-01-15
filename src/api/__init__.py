import logging
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import yaml
import traceback  # Add this

# Configure logging first
logging.basicConfig(
    level=logging.DEBUG)  # Set to DEBUG
logger = logging.getLogger(__name__)

# Load config
try:
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        os.environ["GOOGLE_CLOUD_PROJECT"] = config["gcp"]["project_id"]
        logger.debug("Loaded config successfully")
except Exception as e:
    logger.error(f"Config load error: {e}")
    logger.error(traceback.format_exc())
    raise

# Create FastAPI app
app = FastAPI(
    title="Sheetz Recommendation API",
    description="API for personalized product recommendations",
    version="1.0.0"
)
logger.debug("Created FastAPI app")

# Import and mount router with detailed error handling
try:
    logger.debug("Attempting to import router")
    from .routes import router
    logger.debug("Router imported successfully")
    logger.debug(f"Router routes before mounting: {[r.path for r in router.routes]}")
    
    app.include_router(router)
    logger.debug(f"Router mounted. App routes: {[r.path for r in app.routes]}")
except Exception as e:
    logger.error(f"Router error: {e}")
    logger.error(traceback.format_exc())
    raise

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")
    try:
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    finally:
        logger.info("Shutting down application...")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=True
    )