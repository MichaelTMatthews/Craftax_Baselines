#!/bin/bash

echo 'Building Dockerfile with image name craftax'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt)" \
    -t craftax_baselines \
    --no-cache \
    .
