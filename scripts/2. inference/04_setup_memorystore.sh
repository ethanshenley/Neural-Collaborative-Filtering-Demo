#!/bin/bash
# scripts/inference/04_setup_memorystore.sh

set -e

# Load configuration
PROJECT_ID=$(yq eval '.gcp.project_id' config/config.yaml)
REGION=$(yq eval '.gcp.location' config/config.yaml)
REDIS_INSTANCE="sheetz-rec-cache"
MEMORY_SIZE=$(yq eval '.inference.memorystore.memory_size_gb' config/config.yaml)
REDIS_VERSION="redis_6_x"

echo "Setting up Memorystore (Redis) instance..."

# Create VPC if it doesn't exist
VPC_NAME="sheetz-rec-vpc"
if ! gcloud compute networks describe $VPC_NAME &>/dev/null; then
    echo "Creating VPC network: $VPC_NAME"
    gcloud compute networks create $VPC_NAME \
        --subnet-mode=auto
fi

# Create Memorystore instance
gcloud redis instances create $REDIS_INSTANCE \
    --project=$PROJECT_ID \
    --region=$REGION \
    --zone=${REGION}-a \
    --network=$VPC_NAME \
    --tier=$(yq eval '.inference.memorystore.tier' config/config.yaml) \
    --size=$MEMORY_SIZE \
    --redis-version=$REDIS_VERSION \
    --redis-config maxmemory-policy=allkeys-lru \
    --read-replicas-mode=$(yq eval '.inference.memorystore.read_replicas_mode' config/config.yaml)

# Get instance details
REDIS_HOST=$(gcloud redis instances describe $REDIS_INSTANCE \
    --region=$REGION \
    --format='get(host)')
REDIS_PORT=$(gcloud redis instances describe $REDIS_INSTANCE \
    --region=$REGION \
    --format='get(port)')

# Save Redis configuration
cat > config/redis.yaml << EOL
redis:
  host: $REDIS_HOST
  port: $REDIS_PORT
  instance_id: $REDIS_INSTANCE
  region: $REGION
EOL

# Create Serverless VPC Access connector
CONNECTOR_NAME="sheetz-vpc-connector"
gcloud compute networks vpc-access connectors create $CONNECTOR_NAME \
    --network=$VPC_NAME \
    --region=$REGION \
    --range=10.8.0.0/28

echo "Memorystore setup complete!"
echo "Redis endpoint: $REDIS_HOST:$REDIS_PORT"