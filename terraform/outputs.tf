output "api_url" {
  description = "URL of the deployed API"
  value       = google_cloud_run_service.api.status[0].url
}

output "model_endpoint" {
  description = "Vertex AI model endpoint"
  value       = google_vertex_ai_endpoint.model_endpoint.name
}

output "redis_host" {
  description = "Redis instance hostname"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Redis instance port"
  value       = google_redis_instance.cache.port
}

output "service_account_email" {
  description = "Service account email"
  value       = google_service_account.inference_sa.email
}

output "vpc_connector" {
  description = "VPC Access Connector name"
  value       = google_vpc_access_connector.connector.name
}

output "dashboard_url" {
  description = "URL of the Cloud Monitoring dashboard"
  value       = format("https://console.cloud.google.com/monitoring/dashboards/%s", google_monitoring_dashboard.recommendations.id)
}

output "environment" {
  description = "Deployment environment"
  value       = var.environment
}

output "deployment_time" {
  description = "Time of last deployment"
  value       = timestamp()
}