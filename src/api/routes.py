# src/api/routes.py


from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional
import logging
from datetime import datetime

from src.inference.models import (
    RecommendationRequest,
    RecommendationResponse,
    BatchRecommendationRequest,
    BatchRecommendationResponse,
    ErrorResponse
)
from src.inference.serving import ModelServer
from src.inference.features import FeatureProcessor
from src.inference.cache import FeatureCache
from src.inference.vector_search import ProductSearch

router = APIRouter()

@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks
):
    """Get personalized recommendations for a customer"""
    try:
        start_time = datetime.utcnow()
        
        # Initialize components
        feature_processor = FeatureProcessor()
        model_server = ModelServer()
        feature_cache = FeatureCache()
        product_search = ProductSearch()
        
        # Get user features (with caching)
        features = await feature_cache.get_features(request.customer_id)
        if not features:
            features = await feature_processor.get_features(request.customer_id)
            background_tasks.add_task(
                feature_cache.set_features,
                request.customer_id,
                features
            )
            
        # Get user embedding
        user_embedding = await model_server.get_user_embedding(features)
        
        # Find similar products
        similar_products = await product_search.find_neighbors(
            user_embedding,
            k=request.num_recommendations,
            category_filter=request.category_filter
        )
        
        # Get detailed predictions
        product_ids = [p["product_id"] for p in similar_products]
        predictions = await model_server.get_predictions(
            user_features=features,
            product_ids=product_ids
        )
        
        # Enrich products
        enriched_products = await feature_processor.enrich_products(predictions)
        
        # Calculate inference time
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return RecommendationResponse(
            recommendations=enriched_products,
            metadata={
                "inference_time_ms": inference_time,
                "model_version": model_server.model_version,
                "features_used": list(features.keys()) if request.include_features else None,
                "timestamp": datetime.utcnow()
            }
        )
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post(
    "/recommendations/batch",
    response_model=BatchRecommendationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def batch_recommendations(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks
):
    """Get recommendations for multiple customers"""
    try:
        start_time = datetime.utcnow()
        
        # Initialize components
        feature_processor = FeatureProcessor()
        model_server = ModelServer()
        feature_cache = FeatureCache()
        product_search = ProductSearch()
        
        # Get features for all customers
        all_features = {}
        for customer_id in request.customer_ids:
            # Try cache first
            features = await feature_cache.get_features(customer_id)
            if not features:
                features = await feature_processor.get_features(customer_id)
                background_tasks.add_task(
                    feature_cache.set_features,
                    customer_id,
                    features
                )
            all_features[customer_id] = features
            
        # Get embeddings for all users
        embeddings = {
            customer_id: await model_server.get_user_embedding(features)
            for customer_id, features in all_features.items()
        }
        
        # Find similar products for each user
        all_recommendations = {}
        for customer_id, embedding in embeddings.items():
            similar_products = await product_search.find_neighbors(
                embedding,
                k=request.num_recommendations,
                category_filter=request.category_filter
            )
            
            # Get detailed predictions
            product_ids = [p["product_id"] for p in similar_products]
            predictions = await model_server.get_predictions(
                user_features=all_features[customer_id],
                product_ids=product_ids
            )
            
            # Enrich products
            enriched_products = await feature_processor.enrich_products(predictions)
            all_recommendations[customer_id] = enriched_products
            
        inference_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return BatchRecommendationResponse(
            results=all_recommendations,
            metadata={
                "inference_time_ms": inference_time,
                "model_version": model_server.model_version,
                "timestamp": datetime.utcnow()
            }
        )
        
    except Exception as e:
        logging.error(f"Error generating batch recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@router.get("/metrics")
async def get_metrics():
    """Get inference metrics"""
    try:
        product_search = ProductSearch()
        index_stats = await product_search.get_stats()
        
        return {
            "index_stats": index_stats,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.post("/index/refresh")
async def refresh_index(background_tasks: BackgroundTasks):
    """Refresh vector search index"""
    try:
        product_search = ProductSearch()
        background_tasks.add_task(product_search.refresh_index)
        
        return {
            "status": "refresh_scheduled",
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
