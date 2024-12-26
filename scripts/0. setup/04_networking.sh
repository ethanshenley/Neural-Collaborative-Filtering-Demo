#!/bin/bash
set -e

PROJECT_ID="sheetz-poc"
NETWORK="vertex-network"
SUBNET="vertex-subnet"
REGION="us-central1"
RANGE="10.0.0.0/16"

echo "Setting up networking..."

# Create VPC
gcloud compute networks create $NETWORK \
    --project=$PROJECT_ID \
    --subnet-mode=custom

# Create subnet
gcloud compute networks subnets create $SUBNET \
    --project=$PROJECT_ID \
    --network=$NETWORK \
    --region=$REGION \
    --range=$RANGE \
    --enable-private-ip-google-access

# Set up firewall rules
gcloud compute firewall-rules create allow-internal \
    --project=$PROJECT_ID \
    --network=$NETWORK \
    --allow=tcp,udp,icmp \
    --source-ranges=$RANGE

echo "Network setup complete!"