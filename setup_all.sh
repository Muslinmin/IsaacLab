#!/bin/bash

echo "=========================================="
echo "Setting up complete environment..."
echo "=========================================="
echo ""

# Source conda activation
if [ -f "/workspace/activate_conda.sh" ]; then
    echo "Activating conda environment..."
    source /workspace/activate_conda.sh
else
    echo "Warning: activate_conda.sh not found"
fi

echo ""

# Source Isaac Sim environment
if [ -f "/workspace/setup_isaac_env.sh" ]; then
    echo "Setting up Isaac Sim environment..."
    source /workspace/setup_isaac_env.sh
else
    echo "Warning: setup_isaac_env.sh not found"
fi

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "=========================================="
