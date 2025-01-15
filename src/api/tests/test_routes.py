from fastapi.testclient import TestClient
from src.api import app
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_routes():
    logger.debug("Available routes:")
    for route in app.routes:
        logger.debug(f"Path: {route.path}, Methods: {route.methods}")

    # Test with schema from src/inference/models.py
    response = client.post(
        "/recommendations",
        json={
            "customer_id": "TEST_001",
            "num_recommendations": 5,
            "category_filter": None,  # Optional, so can be None
            "include_features": False  # Optional, defaulting to False
        }
    )
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response body: {response.json()}")