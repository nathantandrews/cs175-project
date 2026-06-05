#!/bin/bash

# Check if we are running inside the Docker container
if [ -f "/.dockerenv" ] || [ "$(pwd)" = "/model" ]; then
    python3 src/main.py --mode test "$@"
    echo "TensorBoard is running. Access it at http://localhost:6006"
    tensorboard --logdir=/model/runs --bind_all 
    

else 
    docker exec -it model python3 src/main.py --mode test "$@"

fi

