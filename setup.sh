#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Step 1: Checking and downloading datasets ==="
./scripts/download-datasets.sh

echo ""
echo "=== Step 2: Checking Docker container ==="

# Check if the 'model' container is already running
if docker ps --format '{{.Names}}' | grep -Eq "^model$"; then
    echo "Docker container 'model' is already running. Setup is complete!"

# Check if 'model' container exists but is stopped
elif docker ps -a --format '{{.Names}}' | grep -Eq "^model$"; then
    echo "Docker container 'model' exists but is stopped. Starting it..."
    docker start model
    echo "Setup is complete!"

# Otherwise, set up Docker from scratch
else
    echo "Docker container 'model' not found. Setting up Docker container..."
    ./scripts/setup-docker.sh
    echo "Setup is complete!"
fi
