#!/bin/bash

docker build -t cs175 .
#docker run  -d --name model -v .:/model cs175 sleep infinity
docker run -d --name model --device nvidia.com/gpu=all   --shm-size=8g   --security-opt label=disable -v .:/model cs175 sleep infinity
