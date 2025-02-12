#!/bin/bash
set -e

PROJECT_ID="sheetz-poc"
SA_NAME="vertex-training"
SA_ID="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "Creating service account for Vertex AI training..."

# Create service account
gcloud iam service-accounts create $SA_NAME \
    --display-name="Vertex AI Training Service Account" \
    --project=$PROJECT_ID

# Grant necessary roles
roles=(
    "roles/aiplatform.user"
    "roles/bigquery.dataViewer"
    "roles/bigquery.jobUser"  # Add this role
    "roles/bigquery.user"     # Add this role
    "roles/storage.objectViewer"
    "roles/monitoring.metricWriter"
    "roles/logging.logWriter"
)

for role in "${roles[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SA_ID" \
        --role="$role"
done

echo "Service account setup complete!"