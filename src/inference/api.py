# src/inference/api.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from google.cloud import aiplatform, bigquery, monitoring
from google.cloud.monitoring_v3 import MetricServiceClient
import asyncio
import time
from typing import List, Optional
from pydantic import BaseModel

from src.inference.features import FeatureProcessor
from src.inference.serving import ModelServer
from src.inference.cache import FeatureCache
from src.inference.vector_search import ProductSearch
from src.inference.monitoring import MetricsLogger

class RecommendationRequest(BaseModel):
    customer_id: str
    num_recommendations: int = 10
    category_filter: Optional[str] = None
    include_features: bool = False

class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    metadata: dict

app = FastAPI(title="Sheetz Recommendation API")
metrics = MetricsLogger()

@app.on_event("startup")
async def startup_event():
    app.state.feature_processor = FeatureProcessor()
    app.state.model_server = ModelServer()
    app.state.feature_cache = FeatureCache()
    app.state.product_search = ProductSearch()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks
):
    start_time = time.time()
    
    try:
        # 1. Get user features (with caching)
        features = await app.state.feature_cache.get_features(
            request.customer_id
        )
        if not features:
            features = await app.state.feature_processor.get_features(
                request.customer_id
            )
            background_tasks.add_task(
                app.state.feature_cache.set_features,
                request.customer_id,
                features
            )

        # 2. Get user embedding from model
        user_embedding = await app.state.model_server.get_user_embedding(
            features
        )

        # 3. Find similar products
        similar_products = await app.state.product_search.find_neighbors(
            user_embedding,
            k=request.num_recommendations,
            category_filter=request.category_filter
        )

        # 4. Process results
        recommendations = await app.state.feature_processor.enrich_products(
            similar_products
        )

        # 5. Log metrics
        inference_time = (time.time() - start_time) * 1000
        background_tasks.add_task(
            metrics.log_latency,
            inference_time
        )

        return RecommendationResponse(
            recommendations=recommendations,
            metadata={
                "inference_time_ms": inference_time,
                "features_used": list(features.keys()) if request.include_features else None,
                "model_version": app.state.model_server.model_version
            }
        )

    except Exception as e:
        metrics.log_error()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recommendations/batch")
async def batch_recommendations(
    customer_ids: List[str],
    background_tasks: BackgroundTasks
):
    """Batch recommendation endpoint for multiple users"""
    results = await asyncio.gather(*[
        get_recommendations(
            RecommendationRequest(customer_id=cid),
            background_tasks
        )
        for cid in customer_ids
    ])
    return {"results": results}