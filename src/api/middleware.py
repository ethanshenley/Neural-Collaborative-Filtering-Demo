# src/api/middleware.py

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from google.cloud import monitoring_v3
import time
import logging
from typing import Callable
import yaml
import json

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and metrics"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Initialize monitoring client
        self.client = monitoring_v3.MetricServiceClient()
        
        # Load config
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
            self.project_name = f"projects/{config['gcp']['project_id']}"
            
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Start timing
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate metrics
            duration_ms = (time.time() - start_time) * 1000
            
            # Log metrics
            self._log_request_metrics(
                duration_ms=duration_ms,
                path=request.url.path,
                method=request.method,
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            # Log error metrics
            self._log_error_metrics(
                error_type=type(e).__name__,
                path=request.url.path
            )
            raise
            
    def _log_request_metrics(
        self,
        duration_ms: float,
        path: str,
        method: str,
        status_code: int
    ):
        """Log request-related metrics"""
        try:
            # Create time series
            series = monitoring_v3.TimeSeries()
            
            # Latency metric
            series.metric.type = "custom.googleapis.com/recommendation/latency"
            series.resource.type = "global"
            series.metric.labels.update({
                "path": path,
                "method": method,
                "status_code": str(status_code)
            })
            
            point = series.points.add()
            point.value.double_value = duration_ms
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
            # Log for debugging
            logging.debug(
                f"Request metrics - Path: {path}, Method: {method}, "
                f"Status: {status_code}, Duration: {duration_ms:.2f}ms"
            )
            
        except Exception as e:
            logging.error(f"Error logging metrics: {str(e)}")
            
    def _log_error_metrics(self, error_type: str, path: str):
        """Log error metrics"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/recommendation/errors"
            series.resource.type = "global"
            series.metric.labels.update({
                "error_type": error_type,
                "path": path
            })
            
            point = series.points.add()
            point.value.int64_value = 1
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
        except Exception as e:
            logging.error(f"Error logging error metrics: {str(e)}")

class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for cache control headers"""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        response = await call_next(request)
        
        # Add cache control headers
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        
        return response

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for request validation and logging"""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Log request
        body = None
        if request.method in ["POST", "PUT"]:
            body = await request.body()
            body = json.loads(body)
            logging.info(
                f"Request: {request.method} {request.url.path}\n"
                f"Body: {json.dumps(body, indent=2)}"
            )
        
        response = await call_next(request)
        
        # Log response for non-200 status codes
        if response.status_code != 200:
            body = response.body.decode()
            logging.warning(
                f"Non-200 Response: {response.status_code}\n"
                f"Body: {body}"
            )
            
        return response