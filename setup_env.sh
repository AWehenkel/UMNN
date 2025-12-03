#!/bin/bash
# Setup script for UMNN development environment

echo "Setting up UMNN development environment..."

# Check if micromamba is installed
if ! command -v micromamba &> /dev/null; then
    echo "Error: micromamba not found. Please install it first:"
    echo "  curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj bin/micromamba"
    echo "  Or visit: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html"
    exit 1
fi

# Create environment
echo "Creating micromamba environment 'umnn-dev'..."
micromamba env create -f environment.yml -y

echo ""
echo "✓ Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  micromamba activate umnn-dev"
echo ""
echo "To test the installation, run:"
echo "  python test_jit.py"
