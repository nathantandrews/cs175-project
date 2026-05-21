#!/bin/bash

# Check if we are running inside the Docker container
if [ -f "/.dockerenv" ] || [ "$(pwd)" = "/model" ]; then
    python3 src/main.py --mode grid_search "$@"
else
    docker exec -it model python3 src/main.py --mode grid_search "$@"
fi