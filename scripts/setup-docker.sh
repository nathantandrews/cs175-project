#!/bin/bash

docker build -t cs175 .
docker run  -d --name model -v .:/model cs175 sleep infinity
