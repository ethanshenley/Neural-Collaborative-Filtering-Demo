# src/inference/cache.py

import redis
import json
import pickle
import logging
from typing import Any, Optional, Dict
import yaml
import asyncio
from google.cloud import monitoring_v3

class FeatureCache:
    """Redis-based caching for feature vectors and embeddings"""
    
    def __init__(self):
        # Load Redis configuration
        with open("config/redis.yaml") as f:
            redis_config = yaml.safe_load(f)["redis"]
            
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=redis_config["host"],
            port=redis_config["port"],
            decode_responses=False  # Keep as bytes for pickle
        )
        
        # Setup monitoring
        self.metric_client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{redis_config['project_id']}"
        
        # Cache configuration
        self.feature_ttl = 3600  # 1 hour
        self.embedding_ttl = 86400  # 24 hours
        
    async def get_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get cached feature vector for a customer"""
        try:
            key = f"features:{customer_id}"
            cached_data = await asyncio.to_thread(
                self.redis.get,
                key
            )
            
            if cached_data:
                # Log cache hit
                self._log_cache_metric(1.0)
                return pickle.loads(cached_data)
            
            # Log cache miss
            self._log_cache_metric(0.0)
            return None
            
        except Exception as e:
            logging.error(f"Cache error for customer {customer_id}: {str(e)}")
            return None
            
    async def set_features(self, customer_id: str, features: Dict[str, Any]) -> bool:
        """Cache feature vector for a customer"""
        try:
            key = f"features:{customer_id}"
            pickled_data = pickle.dumps(features)
            
            success = await asyncio.to_thread(
                self.redis.setex,
                key,
                self.feature_ttl,
                pickled_data
            )
            return bool(success)
            
        except Exception as e:
            logging.error(f"Cache set error for customer {customer_id}: {str(e)}")
            return False
            
    async def get_embedding(self, customer_id: str) -> Optional[bytes]:
        """Get cached user embedding"""
        try:
            key = f"embedding:{customer_id}"
            return await asyncio.to_thread(
                self.redis.get,
                key
            )
        except Exception as e:
            logging.error(f"Embedding cache error for {customer_id}: {str(e)}")
            return None
            
    async def set_embedding(self, customer_id: str, embedding: bytes) -> bool:
        """Cache user embedding"""
        try:
            key = f"embedding:{customer_id}"
            success = await asyncio.to_thread(
                self.redis.setex,
                key,
                self.embedding_ttl,
                embedding
            )
            return bool(success)
        except Exception as e:
            logging.error(f"Embedding cache set error for {customer_id}: {str(e)}")
            return False
            
    def _log_cache_metric(self, hit: float):
        """Log cache hit/miss metric"""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/recommendation/cache_hit_rate"
        series.resource.type = "global"
        
        point = series.points.add()
        point.value.double_value = hit
        
        self.metric_client.create_time_series(
            request={
                "name": self.project_name,
                "time_series": [series]
            }
        )
        
    async def clear_customer_cache(self, customer_id: str):
        """Clear all cached data for a customer"""
        try:
            keys = [
                f"features:{customer_id}",
                f"embedding:{customer_id}"
            ]
            await asyncio.to_thread(self.redis.delete, *keys)
        except Exception as e:
            logging.error(f"Cache clear error for {customer_id}: {str(e)}")