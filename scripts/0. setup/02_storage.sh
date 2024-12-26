#!/bin/bash
set -e

PROJECT_ID="sheetz-poc"
LOCATION="us-central1"
BUCKETS=(
    "sheetz-rec-staging"
    "sheetz-rec-models"
    "sheetz-rec-tensorboard"
)

echo "Setting up GCS buckets..."

for bucket in "${BUCKETS[@]}"; do
    if ! gsutil ls gs://${bucket} &>/dev/null; then
        gsutil mb -p $PROJECT_ID -l $LOCATION gs://${bucket}
        gsutil versioning set on gs://${bucket}
        
        # Set lifecycle policy for training artifacts
        if [[ $bucket == *"staging"* ]]; then
            cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 30,
          "isLive": false
        }
      }
    ]
  }
}
EOF
            gsutil lifecycle set /tmp/lifecycle.json gs://${bucket}
        fi
    fi
done

echo "GCS buckets setup complete!"