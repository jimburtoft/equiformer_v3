#!/bin/bash
set -e

# === PyTorch Native setup script for trn2.3xlarge ===
# Run once on a fresh DLAMI instance.
#
# IMPORTANT: Training (backward pass) ONLY works inside the Docker container.
# The venv approach with host runtime is broken for backward/autograd.
# Use run_in_docker.sh to execute code.

echo "=== Step 1: ECR login and pull DLC ==="
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

echo "=== Step 2: Install model dependencies into a persistent container ==="
imageID=$(docker images -q --filter reference=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest)

# Create a persistent container with deps installed
docker run --name equiformer_v3_env --privileged $imageID bash -c '
pip install e3nn torch-geometric packaging -q
echo "Dependencies installed"
'
# Commit it so we don't reinstall every time
docker commit equiformer_v3_env equiformer_v3:latest
docker rm equiformer_v3_env

echo "=== Setup complete ==="
echo ""
echo "Usage: To run code inside the container:"
echo "  docker run --rm --privileged \\"
echo "    -v /home/ubuntu/code:/work \\"
echo "    -e NEURON_RT_VISIBLE_CORES=0 \\"
echo "    -e HOME=/root \\"
echo "    equiformer_v3:latest \\"
echo "    bash -c 'cd /work && python your_script.py'"
echo ""
echo "Or use: bash run_in_docker.sh your_script.py"
