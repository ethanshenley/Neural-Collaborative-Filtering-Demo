# scripts/setup/setup_terraform.sh

#!/bin/bash
set -e

# Load configuration
PROJECT_ID=$(yq eval '.gcp.project_id' ../config/config.yaml)
REGION=$(yq eval '.gcp.location' ../config/config.yaml)
BUCKET_NAME="sheetz-rec-terraform-state"

echo "Setting up Terraform infrastructure for project: $PROJECT_ID"

# Create GCS bucket for Terraform state
echo "Creating Terraform state bucket..."
gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_NAME

# Enable versioning on the bucket
echo "Enabling versioning on state bucket..."
gsutil versioning set on gs://$BUCKET_NAME

# Create prod.tfvars if it doesn't exist
if [ ! -f "../terraform/prod.tfvars" ]; then
    echo "Creating prod.tfvars..."
    cat > ../terraform/prod.tfvars << EOL
project_id = "$PROJECT_ID"
region = "$REGION"
alert_email = "alerts@sheetz-rec.com"
environment = "production"
model_version = "latest"
model_artifact_bucket = "sheetz-rec-model-artifacts"
domain = "api.sheetz-rec.com"

tags = {
  environment = "production"
  managed_by = "terraform"
  project = "sheetz-rec"
}
EOL
fi

# Create backend config
echo "Creating backend configuration..."
cat > ../terraform/backend.tf << EOL
terraform {
  backend "gcs" {
    bucket = "$BUCKET_NAME"
    prefix = "terraform/state"
  }
}
EOL

echo "Terraform setup complete! You can now run:"
echo "cd ../terraform"
echo "terraform init"
echo "terraform plan -var-file=prod.tfvars"