# scripts/monitoring/create_dashboards.py

from google.cloud import monitoring_v3
from google.cloud.monitoring_dashboards_v1 import DashboardsClient
import yaml
import json

def create_recommendation_dashboard(project_id: str):
    """Create a Cloud Monitoring dashboard for recommendation metrics"""
    client = DashboardsClient()
    
    dashboard = {
        "displayName": "Recommendation System Dashboard",
        "gridLayout": {
            "columns": "2",
            "widgets": [
                # Latency Widget
                {
                    "title": "Inference Latency",
                    "xyChart": {
                        "dataSets": [{
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": 'metric.type="custom.googleapis.com/recommendation/inference_latency"'
                                },
                                "aggregation": {
                                    "alignmentPeriod": "60s",
                                    "crossSeriesReducer": "REDUCE_MEAN",
                                    "perSeriesAligner": "ALIGN_MEAN"
                                }
                            }
                        }]
                    }
                },
                # Cache Hit Rate Widget
                {
                    "title": "Cache Hit Rate",
                    "xyChart": {
                        "dataSets": [{
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": 'metric.type="custom.googleapis.com/recommendation/cache_hit_rate"'
                                },
                                "aggregation": {
                                    "alignmentPeriod": "60s",
                                    "perSeriesAligner": "ALIGN_MEAN"
                                }
                            }
                        }]
                    }
                },
                # Request Rate Widget
                {
                    "title": "Requests per Second",
                    "xyChart": {
                        "dataSets": [{
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": 'metric.type="custom.googleapis.com/recommendation/requests_per_second"'
                                },
                                "aggregation": {
                                    "alignmentPeriod": "60s",
                                    "perSeriesAligner": "ALIGN_RATE"
                                }
                            }
                        }]
                    }
                },
                # Error Rate Widget
                {
                    "title": "Error Rate",
                    "xyChart": {
                        "dataSets": [{
                            "timeSeriesQuery": {
                                "timeSeriesFilter": {
                                    "filter": 'metric.type="custom.googleapis.com/recommendation/error_rate"'
                                },
                                "aggregation": {
                                    "alignmentPeriod": "60s",
                                    "perSeriesAligner": "ALIGN_RATE"
                                }
                            }
                        }]
                    }
                }
            ]
        }
    }
    
    dashboard_path = f"projects/{project_id}"
    response = client.create_dashboard(
        parent=dashboard_path,
        dashboard=dashboard
    )
    
    print(f"Created dashboard: {response.name}")
    return response

def main():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
        
    project_id = config['gcp']['project_id']
    
    # Create dashboards
    recommendation_dashboard = create_recommendation_dashboard(project_id)
    
    # Save dashboard configurations
    dashboard_config = {
        "dashboards": {
            "recommendation": recommendation_dashboard.name
        }
    }
    
    with open("config/dashboards.yaml", "w") as f:
        yaml.dump(dashboard_config, f)

if __name__ == "__main__":
    main()