#!/bin/bash
set -e

# === PyTorch Native setup script for trn2.3xlarge ===
# Run once on a fresh DLAMI instance.
#
# IMPORTANT: Training (backward pass) ONLY works inside the Docker container.
# The venv approach with host runtime is broken for backward/autograd.
# Use run_in_docker.sh to execute code.

# Set NEURON_PTN_IMAGE to the public PyTorch Native beta image URI and tag
# (see the AWS Neuron documentation), e.g.:
#   export NEURON_PTN_IMAGE=<public-pytorch-native-beta-image>:<tag>
: "${NEURON_PTN_IMAGE:?Set NEURON_PTN_IMAGE to the public PyTorch Native beta image}"

echo "=== Step 1: Pull the PyTorch Native image ==="
docker pull "$NEURON_PTN_IMAGE"

echo "=== Step 2: Install model dependencies into a persistent container ==="
imageID=$(docker images -q --filter reference="$NEURON_PTN_IMAGE")

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
