#!/bin/bash
set -e

PROJECT_ID="sheetz-poc"
LOCATION="us-central1"
REPOSITORY="sheetz-training"

echo "Setting up Artifact Registry..."

# Create repository
gcloud artifacts repositories create $REPOSITORY \
    --repository-format=docker \
    --location=$LOCATION \
    --project=$PROJECT_ID

# Configure Docker auth
gcloud auth configure-docker ${LOCATION}-docker.pkg.dev

echo "Artifact Registry setup complete!"