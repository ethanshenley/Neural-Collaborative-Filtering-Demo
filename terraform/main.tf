terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  backend "gcs" {
    bucket = "sheetz-rec-terraform-state"
    prefix = "inference"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "sheetz-rec-vpc"
  auto_create_subnetworks = true

  depends_on = [
    google_project_service.required_apis
  ]
}

# VPC Access Connector
resource "google_vpc_access_connector" "connector" {
  name          = "sheetz-vpc-connector"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = var.vpc_connector_range  # This is defined in your variables.tf as "10.8.0.0/28"
  
  # Machine type for the connector
  machine_type = "e2-micro"  # Use smallest machine type to minimize costs
  
  # Minimum and maximum instances
  min_instances = 2
  max_instances = 3
  
  # Add dependency on API enablement
  depends_on = [
    google_project_service.required_apis
  ]
}

# Cloud Run service
resource "google_cloud_run_service" "api" {
  name     = "sheetz-rec-api"
  location = var.region

  template {
    spec {
      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/sheetz-rec/api:latest"
        
        resources {
          limits = {
            cpu    = "2"
            memory = "2Gi"
          }
        }
        
        env {
          name  = "PROJECT_ID"
          value = var.project_id
        }
        
        env {
          name  = "REGION"
          value = var.region
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.connector.name
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

# Vertex AI endpoint
resource "google_vertex_ai_endpoint" "model_endpoint" {
  name         = "sheetz-rec-endpoint"
  description  = "Recommendation model endpoint"
  location     = var.region
  display_name = "Recommendation System"

  network = "projects/${var.project_id}/global/networks/${google_compute_network.vpc.name}"
  
  # Add dependency on API enablement
  depends_on = [
    google_project_service.required_apis
  ]

  network = format(
    "projects/%s/global/networks/%s",
    var.project_id,
    google_compute_network.vpc.name
  )
}

# Memorystore (Redis)
resource "google_redis_instance" "cache" {
  name           = "sheetz-rec-cache"
  tier           = "BASIC"
  memory_size_gb = 5
  region         = var.region

  authorized_network = google_compute_network.vpc.id
  
  redis_version     = "REDIS_6_X"
  display_name      = "Recommendation Cache"
  
  depends_on = [
    google_project_service.required_apis
  ]

  depends_on = [time_sleep.api_enable]
}

# Cloud Monitoring Dashboard
resource "google_monitoring_dashboard" "recommendations" {
  dashboard_json = jsonencode({
    displayName = "Recommendation System Dashboard"
    gridLayout = {
      columns = "2"
      widgets = [
        {
          title = "Inference Latency"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/recommendation/inference_latency\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    crossSeriesReducer = "REDUCE_MEAN"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        },
        {
          title = "Cache Hit Rate"
          xyChart = {
            dataSets = [{
              timeSeriesQuery = {
                timeSeriesFilter = {
                  filter = "metric.type=\"custom.googleapis.com/recommendation/cache_hit_rate\" resource.type=\"global\""
                  aggregation = {
                    alignmentPeriod = "60s"
                    perSeriesAligner = "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        }
      ]
    }
  })
}

# Alerting Policy
resource "google_monitoring_alert_policy" "latency_alert" {
  display_name = "High Inference Latency"
  combiner     = "OR"

  conditions {
    display_name = "High latency"
    condition_threshold {
      filter          = "metric.type=\"custom.googleapis.com/recommendation/inference_latency\" AND resource.type=\"global\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 1000
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

# Notification Channel
resource "google_monitoring_notification_channel" "email" {
  display_name = "Recommendation Alerts"
  type         = "email"
  
  labels = {
    email_address = var.alert_email
  }
}

# IAM role for service account
resource "google_service_account" "inference_sa" {
  account_id   = "inference-service"
  display_name = "Inference Service Account"
}

resource "google_project_iam_member" "inference_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataViewer",
    "roles/redis.viewer",
    "roles/monitoring.metricWriter"
  ])
  
  role    = each.key
  member  = "serviceAccount:${google_service_account.inference_sa.email}"
  project = var.project_id
}