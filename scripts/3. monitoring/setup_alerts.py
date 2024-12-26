# scripts/monitoring/setup_alerts.py

from google.cloud import monitoring_v3
import yaml

def create_alert_policy(project_id: str, display_name: str, filter_str: str, 
                       threshold_value: float, duration: int):
    """Create a Cloud Monitoring alert policy"""
    client = monitoring_v3.AlertPolicyServiceClient()
    
    # Alert condition
    condition = monitoring_v3.AlertPolicy.Condition(
        display_name=f"{display_name} Alert Condition",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter=filter_str,
            duration={"seconds": duration},
            comparison="COMPARISON_GT",
            threshold_value=threshold_value,
            trigger={"count": 1}
        )
    )
    
    # Notification channels
    notification_client = monitoring_v3.NotificationChannelServiceClient()
    parent = f"projects/{project_id}"
    
    # Create email notification channel
    email_channel = notification_client.create_notification_channel(
        request={
            "parent": parent,
            "notification_channel": {
                "type": "email",
                "display_name": f"{display_name} Email Notification",
                "labels": {
                    "email_address": "alerts@sheetz-rec.com"
                }
            }
        }
    )
    
    # Alert policy
    alert_policy = monitoring_v3.AlertPolicy(
        display_name=display_name,
        conditions=[condition],
        notification_channels=[email_channel.name],
        alert_strategy=monitoring_v3.AlertPolicy.AlertStrategy(
            auto_close="3600s"  # Auto-close after 1 hour
        )
    )
    
    # Create the alert policy
    client.create_alert_policy(
        request={
            "name": parent,
            "alert_policy": alert_policy
        }
    )

def setup_monitoring_alerts(project_id: str):
    """Setup all monitoring alerts"""
    
    # 1. High Latency Alert
    create_alert_policy(
        project_id=project_id,
        display_name="High Inference Latency",
        filter_str='metric.type="custom.googleapis.com/recommendation/inference_latency"',
        threshold_value=1000.0,  # 1 second
        duration=300  # 5 minutes
    )
    
    # 2. Error Rate Alert
    create_alert_policy(
        project_id=project_id,
        display_name="High Error Rate",
        filter_str='metric.type="custom.googleapis.com/recommendation/error_rate"',
        threshold_value=0.05,  # 5% error rate
        duration=300
    )
    
    # 3. Low Cache Hit Rate Alert
    create_alert_policy(
        project_id=project_id,
        display_name="Low Cache Hit Rate",
        filter_str='metric.type="custom.googleapis.com/recommendation/cache_hit_rate"',
        threshold_value=0.5,  # Below 50% hit rate
        duration=600  # 10 minutes
    )
    
    # 4. High Request Rate Alert
    create_alert_policy(
        project_id=project_id,
        display_name="High Request Rate",
        filter_str='metric.type="custom.googleapis.com/recommendation/requests_per_second"',
        threshold_value=1000.0,  # 1000 RPS
        duration=300
    )

# scripts/monitoring/setup_alerts.py (continued)

def main():
    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    project_id = config['gcp']['project_id']
    
    # Create monitoring metrics
    client = monitoring_v3.MetricServiceClient()
    
    # Setup metric descriptors
    metric_types = {
        "inference_latency": {
            "display_name": "Recommendation Inference Latency",
            "description": "Latency of recommendation inference requests",
            "unit": "ms",
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE
        },
        "cache_hit_rate": {
            "display_name": "Cache Hit Rate",
            "description": "Rate of cache hits for feature retrieval",
            "unit": "1",
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE
        },
        "error_rate": {
            "display_name": "Error Rate",
            "description": "Rate of failed recommendation requests",
            "unit": "1",
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE
        },
        "requests_per_second": {
            "display_name": "Requests Per Second",
            "description": "Number of recommendation requests per second",
            "unit": "1/s",
            "value_type": monitoring_v3.MetricDescriptor.ValueType.DOUBLE,
            "metric_kind": monitoring_v3.MetricDescriptor.MetricKind.GAUGE
        }
    }
    
    for metric_id, metric_info in metric_types.items():
        descriptor = monitoring_v3.MetricDescriptor(
            type_=f"custom.googleapis.com/recommendation/{metric_id}",
            display_name=metric_info["display_name"],
            description=metric_info["description"],
            unit=metric_info["unit"],
            value_type=metric_info["value_type"],
            metric_kind=metric_info["metric_kind"],
            labels=[
                monitoring_v3.LabelDescriptor(
                    key="environment",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                    description="Deployment environment"
                ),
                monitoring_v3.LabelDescriptor(
                    key="model_version",
                    value_type=monitoring_v3.LabelDescriptor.ValueType.STRING,
                    description="Model version"
                )
            ]
        )
        
        project_name = f"projects/{project_id}"
        client.create_metric_descriptor(
            name=project_name,
            metric_descriptor=descriptor
        )
        print(f"Created metric descriptor: {metric_id}")
    
    # Setup alerts
    print("Setting up monitoring alerts...")
    setup_monitoring_alerts(project_id)
    
    # Create uptime check
    uptime_client = monitoring_v3.UptimeCheckServiceClient()
    api_url = f"https://sheetz-rec-api-{project_id}.a.run.app/health"
    
    uptime_check = monitoring_v3.UptimeCheckConfig(
        display_name="Recommendation API Health Check",
        monitored_resource={
            "type": "uptime_url",
            "labels": {
                "project_id": project_id,
                "host": api_url
            }
        },
        http_check={
            "path": "/health",
            "port": 443,
            "use_ssl": True,
            "validate_ssl": True
        },
        period={"seconds": 300}  # Check every 5 minutes
    )
    
    uptime_client.create_uptime_check_config(
        parent=f"projects/{project_id}",
        uptime_check_config=uptime_check
    )
    print("Created uptime check for API health")
    
    print("Alert setup complete!")

if __name__ == "__main__":
    main()