resource "google_project_service" "required_apis" {
  for_each = toset([
    "redis.googleapis.com",
    "vpcaccess.googleapis.com",
    "aiplatform.googleapis.com",
    "monitoring.googleapis.com",
    "compute.googleapis.com",  # Added for VPC network
    "run.googleapis.com"      # Added for Cloud Run
  ])
  
  project = var.project_id
  service = each.key
  
  disable_dependent_services = false
  disable_on_destroy        = false
}

resource "time_sleep" "api_enable" {
  depends_on = [google_project_service.required_apis]
  create_duration = "90s"
}
