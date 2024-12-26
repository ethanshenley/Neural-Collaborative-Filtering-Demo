# terraform/variables.tf

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "alert_email" {
  description = "Email address for monitoring alerts"
  type        = string
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "min_replicas" {
  description = "Minimum number of replicas for Cloud Run"
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of replicas for Cloud Run"
  type        = number
  default     = 10
}

variable "memory_size_gb" {
  description = "Memory size for Redis instance in GB"
  type        = number
  default     = 5
}

variable "vertex_machine_type" {
  description = "Machine type for Vertex AI endpoint"
  type        = string
  default     = "n1-standard-4"
}

variable "model_version" {
  description = "Version of the recommendation model"
  type        = string
}

variable "model_artifact_bucket" {
  description = "GCS bucket containing model artifacts"
  type        = string
}

variable "enable_monitoring" {
  description = "Enable Cloud Monitoring and alerting"
  type        = bool
  default     = true
}

variable "cache_ttl_hours" {
  description = "TTL for cached features in hours"
  type        = number
  default     = 24
}

variable "vpc_connector_range" {
  description = "IP range for VPC Access Connector"
  type        = string
  default     = "10.8.0.0/28"
}

variable "domain" {
  description = "Domain for the API endpoint"
  type        = string
  default     = ""
}

variable "tags" {
  description = "Tags to apply to resources"
  type        = map(string)
  default     = {}
}