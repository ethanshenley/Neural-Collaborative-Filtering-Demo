# src/inference/monitoring.py

from google.cloud import monitoring_v3
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import time
import yaml

class MetricsLogger:
    """Handles logging metrics to Cloud Monitoring"""
    
    def __init__(self):
        # Load config
        with open("config/config.yaml") as f:
            config = yaml.safe_load(f)
            
        self.project_name = f"projects/{config['gcp']['project_id']}"
        self.client = monitoring_v3.MetricServiceClient()
        
    def log_latency(self, duration_ms: float, labels: Optional[Dict[str, str]] = None):
        """Log inference latency metric"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/recommendation/inference_latency"
            series.resource.type = "global"
            
            if labels:
                series.metric.labels.update(labels)
                
            point = series.points.add()
            point.value.double_value = duration_ms
            point.interval.end_time.seconds = int(time.time())
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
        except Exception as e:
            logging.error(f"Error logging latency metric: {str(e)}")
            
    def log_cache_hit(self, hit: bool, labels: Optional[Dict[str, str]] = None):
        """Log cache hit/miss metric"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/recommendation/cache_hit_rate"
            series.resource.type = "global"
            
            if labels:
                series.metric.labels.update(labels)
                
            point = series.points.add()
            point.value.double_value = 1.0 if hit else 0.0
            point.interval.end_time.seconds = int(time.time())
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
        except Exception as e:
            logging.error(f"Error logging cache metric: {str(e)}")
            
    def log_request_count(self, count: int = 1, labels: Optional[Dict[str, str]] = None):
        """Log request count metric"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/recommendation/requests"
            series.resource.type = "global"
            
            if labels:
                series.metric.labels.update(labels)
                
            point = series.points.add()
            point.value.int64_value = count
            point.interval.end_time.seconds = int(time.time())
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
        except Exception as e:
            logging.error(f"Error logging request metric: {str(e)}")
            
    def log_error(self, error_type: str = "unknown", labels: Optional[Dict[str, str]] = None):
        """Log error metric"""
        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = "custom.googleapis.com/recommendation/errors"
            series.resource.type = "global"
            
            labels = labels or {}
            labels["error_type"] = error_type
            series.metric.labels.update(labels)
            
            point = series.points.add()
            point.value.int64_value = 1
            point.interval.end_time.seconds = int(time.time())
            
            self.client.create_time_series(
                request={
                    "name": self.project_name,
                    "time_series": [series]
                }
            )
            
        except Exception as e:
            logging.error(f"Error logging error metric: {str(e)}")
            
    def log_prediction_stats(self, stats: Dict[str, float], labels: Optional[Dict[str, str]] = None):
        """Log prediction performance metrics"""
        try:
            for metric_name, value in stats.items():
                series = monitoring_v3.TimeSeries()
                series.metric.type = f"custom.googleapis.com/recommendation/predictions/{metric_name}"
                series.resource.type = "global"
                
                if labels:
                    series.metric.labels.update(labels)
                    
                point = series.points.add()
                point.value.double_value = value
                point.interval.end_time.seconds = int(time.time())
                
                self.client.create_time_series(
                    request={
                        "name": self.project_name,
                        "time_series": [series]
                    }
                )
                
        except Exception as e:
            logging.error(f"Error logging prediction stats: {str(e)}")

def log_inference_latency(latency_ms):
    try:
        client = monitoring_v3.MetricServiceClient()
        project_path = f"projects/sheetz-poc"
        
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/recommendation/inference_latency"
        series.resource.type = "cloud_run_revision"
        series.metric.labels["service_name"] = "sheetz-rec-api"
        
        point = monitoring_v3.Point()
        point.value.double_value = latency_ms
        series.points = [point]
        
        client.create_time_series(name=project_path, time_series=[series])
        logging.info(f"Successfully logged latency metric: {latency_ms}ms")
    except Exception as e:
        logging.error(f"Failed to log latency metric: {str(e)}")