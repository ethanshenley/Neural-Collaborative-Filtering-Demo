#!/bin/bash
# scripts/inference/02_deploy_model.sh

set -e

# Load configuration
PROJECT_ID=$(yq eval '.gcp.project_id' config/config.yaml)
REGION=$(yq eval '.gcp.location' config/config.yaml)

# Build the model server container
echo "Building model server container..."
docker build -f Dockerfile.serve -t gcr.io/$PROJECT_ID/model-server .

# Push to Container Registry
echo "Pushing container to GCR..."
docker push gcr.io/$PROJECT_ID/model-server

# Copy model artifacts to GCS
echo "Copying model artifacts to GCS..."
MODEL_DIR="gs://$PROJECT_ID-model-artifacts/models/latest"
gsutil -m cp -r ./models/* $MODEL_DIR/

# Create model serving Dockerfile if it doesn't exist
if [ ! -f Dockerfile.serve ]; then
  cat > Dockerfile.serve << EOL
FROM pytorch/torchserve:latest

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model artifacts
COPY models/model.pt /models/
COPY src/inference/serving.py /app/
COPY config/config.yaml /app/

# Set up model server
WORKDIR /app
ENV MODEL_PATH=/models/model.pt
ENV CONFIG_PATH=/app/config.yaml

# Run the model server
CMD ["python", "serving.py"]
EOL
fi

echo "Model deployment configuration complete!"