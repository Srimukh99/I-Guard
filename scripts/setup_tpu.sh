#!/bin/bash
# I-Guard TPU Setup Script
# Automates the installation of dependencies for Google TPU acceleration

set -e

echo "🚀 Starting I-Guard TPU Setup..."

# Update system
sudo apt-get update

# Install basic dependencies
sudo apt-get install -y python3-pip python3-dev

# Install JAX with TPU support
echo "📦 Installing JAX for TPU..."
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install PyTorch XLA
echo "📦 Installing torch_xla..."
pip install torch-xla

# Install Transformers and acceleration tools
echo "📦 Installing transformers and accelerate..."
pip install transformers accelerate sentencepiece

# Install I-Guard requirements
if [ -f "requirements.txt" ]; then
    echo "📦 Installing I-Guard requirements..."
    pip install -r requirements.txt
fi

# Configure environment variables
echo "⚙️ Configuring environment variables..."
export XLA_USE_BF16=1
export PJRT_DEVICE=TPU

# Check installation
python3 -c "import jax; print('JAX version:', jax.__version__); print('TPU detected:', jax.devices())" || echo "⚠️ TPU check failed. This is expected if not running on a TPU VM."

echo "✅ I-Guard TPU Setup Complete!"
echo "Note: If running on a TPU VM, ensure PJRT_DEVICE=TPU is set."
