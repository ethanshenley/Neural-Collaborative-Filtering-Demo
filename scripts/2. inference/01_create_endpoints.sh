#!/bin/bash
# scripts/inference/01_create_endpoints.sh

set -e

# Load configuration
PROJECT_ID=$(yq eval '.gcp.project_id' config/config.yaml)
REGION=$(yq eval '.gcp.location' config/config.yaml)
ENDPOINT_NAME=$(yq eval '.inference.vertex_ai.endpoint_name' config/config.yaml)
MODEL_NAME="sheetz_ncf_model"

echo "Creating Vertex AI endpoint for model serving..."

# Create model endpoint
gcloud ai endpoints create \
  --project=$PROJECT_ID \
  --region=$REGION \
  --display-name=$ENDPOINT_NAME

# Get the endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=$REGION \
  --format='value(ENDPOINT_ID)' \
  --filter="display_name=$ENDPOINT_NAME")

echo "Created endpoint with ID: $ENDPOINT_ID"

# Upload model to Vertex AI
ARTIFACT_URI="gs://$PROJECT_ID-model-artifacts/models/latest"
MODEL_ID=$(gcloud ai models upload \
  --region=$REGION \
  --display-name=$MODEL_NAME \
  --artifact-uri=$ARTIFACT_URI \
  --container-image-uri="gcr.io/$PROJECT_ID/model-server" \
  --format='value(MODEL_ID)')

echo "Uploaded model with ID: $MODEL_ID"

# Deploy model to endpoint
gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=$REGION \
  --model=$MODEL_ID \
  --display-name=$MODEL_NAME \
  --machine-type=$(yq eval '.inference.vertex_ai.machine_type' config/config.yaml) \
  --min-replica-count=$(yq eval '.inference.vertex_ai.min_replica_count' config/config.yaml) \
  --max-replica-count=$(yq eval '.inference.vertex_ai.max_replica_count' config/config.yaml) \
  --traffic-split=0=100

echo "Model deployed successfully to endpoint: $ENDPOINT_ID"

# Save endpoint configuration
cat > config/endpoints.yaml << EOL
vertex_ai:
  endpoint_id: $ENDPOINT_ID
  model_id: $MODEL_ID
  region: $REGION
  project_id: $PROJECT_ID
EOL

echo "Endpoint configuration saved to config/endpoints.yaml"