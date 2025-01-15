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

# Data source for project number
data "google_project" "current" {}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = "sheetz-rec-vpc"
  auto_create_subnetworks = true
  depends_on = [google_project_service.required_apis]
}

# VPC Access Connector
resource "google_vpc_access_connector" "connector" {
  name          = "sheetz-vpc-connector"
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = var.vpc_connector_range
  machine_type  = "e2-micro"
  min_instances = 2
  max_instances = 3
  depends_on    = [google_project_service.required_apis]
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

        startup_probe {
          initial_delay_seconds = 30
          period_seconds = 10
          failure_threshold = 3
          http_get {
            path = "/health"
          }
        }

        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }

        env {
          name = "MODEL_BUCKET"
          value = var.model_artifact_bucket
        }
        env {
          name = "MODEL_VERSION"
          value = var.model_version
        }
        env {
          name = "VERTEX_ENDPOINT_ID"
          value = google_vertex_ai_endpoint.model_endpoint.name
        }
      }
    }

    metadata {
      annotations = {
        "autoscaling.knative.dev/minScale" = "1"
        "autoscaling.knative.dev/maxScale" = "10"
        "run.googleapis.com/startup-cpu-boost" = "true"
      }
    }
  }
}

# Vertex AI endpoint
resource "google_vertex_ai_endpoint" "model_endpoint" {
  name         = "sheetz-rec-endpoint"
  description  = "Recommendation model endpoint"
  location     = var.region
  display_name = "Recommendation System"

  network = format(
    "projects/%s/global/networks/%s",
    data.google_project.current.number,
    google_compute_network.vpc.name
  )
  
  depends_on = [
    google_project_service.required_apis,
    time_sleep.api_enable
  ]
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
    time_sleep.api_enable,
    google_compute_network.vpc
  ]
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

  depends_on = [
    google_project_service.required_apis
  ]
}
# Service Account
resource "google_service_account" "inference_sa" {
  account_id   = "inference-service"
  display_name = "Inference Service Account"
}

# IAM roles
resource "google_project_iam_member" "inference_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/bigquery.dataViewer",
    "roles/redis.viewer",
    "roles/monitoring.metricWriter",
    "roles/run.invoker",
    "roles/run.serviceAgent",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter"
  ])
  
  role    = each.key
  member  = "serviceAccount:${google_service_account.inference_sa.email}"
  project = var.project_id
}

# Allow Cloud Run to use the service account
resource "google_service_account_iam_member" "cloud_run_service_account" {
  service_account_id = google_service_account.inference_sa.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${data.google_project.current.number}-compute@developer.gserviceaccount.com"
}