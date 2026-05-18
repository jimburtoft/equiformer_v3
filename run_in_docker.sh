#!/bin/bash
# Run a Python script inside the PyTorch Native Docker container.
# Usage: bash run_in_docker.sh <script.py> [args...]
#
# The script is expected to be in ~/code/ on the host.
# The equiformer_v3 model package should be at ~/code/equiformer_v3/

SCRIPT=$1
shift

docker run --rm --privileged \
  -v /home/ubuntu/code:/root/equiformer_v3 \
  -v /home/ubuntu/code/$SCRIPT:/root/$SCRIPT \
  -e NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-0} \
  -e HOME=/root \
  equiformer_v3:latest \
  bash -c "cd /root && python $SCRIPT $*"
