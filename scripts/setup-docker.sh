#!/bin/bash

# Build the Docker image
docker build -t cs175 .

# Check if the build was successful
if [ $? -eq 0 ]; then
    # Prompt the user for GPU usage
    read -p "Do you want to use an NVIDIA GPU? (y/n): " use_gpu

    if [[ "$use_gpu" =~ ^[Yy]$ ]]; then
        echo "Running with GPU support..."
        docker run -p 6006:6006 -d --name model --device nvidia.com/gpu=all --shm-size=8g --security-opt label=disable -v .:/model cs175 sleep infinity
    else
        echo "Running without GPU support (standard mode)..."
        docker run -p 6006:6006 -d --name model -v .:/model cs175 sleep infinity
    fi
else
    echo "Docker build failed. Exiting."
    exit 1
fi