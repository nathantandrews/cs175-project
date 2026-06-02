#!/bin/bash

echo "Starting TensorBoard..."
docker exec -d model tensorboard --logdir=/model/runs --bind_all
echo "TensorBoard is running. Access it at http://localhost:6006"
echo "To stop TensorBoard, run: docker exec -it model pkill tensorboard"
echo "Launching interactive container shell..."
docker exec -it model bash

