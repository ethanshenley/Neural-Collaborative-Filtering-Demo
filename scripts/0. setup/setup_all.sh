#!/bin/bash
set -e

# Run all setup scripts in order
echo "Starting complete GCP infrastructure setup..."

scripts=(
    "01_service_account.sh"
    "02_storage.sh"
    "03_artifact_registry.sh"
    "04_networking.sh"
)

for script in "${scripts[@]}"; do
    echo "Running $script..."
    bash "scripts/setup/$script"
    echo "Completed $script"
    echo "-------------------"
done

echo "All infrastructure setup complete!"
