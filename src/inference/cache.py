# src/inference/cache.py

import redis
import json
import pickle
import logging
from typing import Any, Optional, Dict
import yaml
import asyncio
from google.cloud import monitoring_v3
from collections import defaultdict
import time

class DummyCache:
    """In-memory cache fallback when Redis is not available"""
    def __init__(self):
        self.features = {}
        self.embeddings = {}
        self.feature_ttl = 3600  # 1 hour
        self.embedding_ttl = 86400  # 24 hours
        self.timestamps = defaultdict(dict)
        logging.warning("Using in-memory cache - NOT FOR PRODUCTION")
        
    async def get_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        key = f"features:{customer_id}"
        if key in self.features:
            # Check TTL
            if time.time() - self.timestamps[key].get('set_time', 0) < self.feature_ttl:
                return self.features[key]
            else:
                del self.features[key]
        return None
        
    async def set_features(self, customer_id: str, features: Dict[str, Any]) -> bool:
        try:
            key = f"features:{customer_id}"
            self.features[key] = features
            self.timestamps[key]['set_time'] = time.time()
            return True
        except Exception:
            return False
            
    async def get_embedding(self, customer_id: str) -> Optional[bytes]:
        key = f"embedding:{customer_id}"
        if key in self.embeddings:
            if time.time() - self.timestamps[key].get('set_time', 0) < self.embedding_ttl:
                return self.embeddings[key]
            else:
                del self.embeddings[key]
        return None
        
    async def set_embedding(self, customer_id: str, embedding: bytes) -> bool:
        try:
            key = f"embedding:{customer_id}"
            self.embeddings[key] = embedding
            self.timestamps[key]['set_time'] = time.time()
            return True
        except Exception:
            return False
            
    async def clear_customer_cache(self, customer_id: str):
        keys = [f"features:{customer_id}", f"embedding:{customer_id}"]
        for key in keys:
            self.features.pop(key, None)
            self.embeddings.pop(key, None)
            self.timestamps.pop(key, None)

class FeatureCache:
    """Redis-based caching for feature vectors and embeddings"""
    
    def __init__(self):
        try:
            # Load Redis configuration
            with open("config/redis.yaml") as f:
                redis_config = yaml.safe_load(f)["redis"]
                
            # Initialize Redis connection
            self.redis = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                decode_responses=False  # Keep as bytes for pickle
            )
            
            # Setup monitoring - make it optional
            try:
                self.metric_client = monitoring_v3.MetricServiceClient()
                self.project_name = f"projects/{redis_config.get('project_id', 'sheetz-poc')}"
            except Exception as e:
                logging.warning(f"Monitoring setup failed: {e}")
                self.metric_client = None
                self.project_name = None
                
            # Test Redis connection
            self.redis.ping()
            logging.info("Redis cache initialized successfully")
            
        except (FileNotFoundError, redis.ConnectionError) as e:
            logging.warning(f"Redis not available ({str(e)}), falling back to in-memory cache")
            # Use DummyCache as fallback
            self._fallback = DummyCache()
            self.redis = None
            self.metric_client = None
            
        # Cache configuration
        self.feature_ttl = 3600  # 1 hour
        self.embedding_ttl = 86400  # 24 hours
        
    async def get_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get cached feature vector for a customer"""
        if not self.redis:
            return await self._fallback.get_features(customer_id)
            
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
        if not self.metric_client or not self.project_name:
            return
            
        try:
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
        except Exception as e:
            logging.warning(f"Failed to log metric: {e}")

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