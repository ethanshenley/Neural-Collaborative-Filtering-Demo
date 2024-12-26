#!/bin/bash
# scripts/inference/setup_inference.sh

set -e

# Load configuration
PROJECT_ID=$(yq eval '.gcp.project_id' config/config.yaml)
REGION=$(yq eval '.gcp.location' config/config.yaml)
MODEL_ID=$(yq eval '.inference.vertex_ai.endpoint_name' config/config.yaml)

echo "Setting up inference infrastructure for project: $PROJECT_ID"

# 1. Deploy model to Vertex AI
echo "Deploying model to Vertex AI..."
./01_create_endpoints.sh

# 2. Create Vector Search Index
echo "Setting up Vector Search..."
./02_setup_vector_search.sh

# 3. Setup Memorystore
echo "Setting up Memorystore..."
./03_setup_memorystore.sh

# 4. Deploy Cloud Run API
echo "Deploying API to Cloud Run..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/sheetz-rec-api
gcloud run deploy sheetz-rec-api \
    --image gcr.io/$PROJECT_ID/sheetz-rec-api \
    --platform managed \
    --region $REGION \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 1 \
    --max-instances 10 \
    --timeout 30s \
    --set-env-vars PROJECT_ID=$PROJECT_ID,REGION=$REGION,MODEL_ID=$MODEL_ID

# 5. Setup monitoring
echo "Setting up monitoring..."
python scripts/monitoring/setup_alerts.py

echo "Inference infrastructure setup complete!"
echo "API URL: $(gcloud run services describe sheetz-rec-api --format='value(status.url)')"