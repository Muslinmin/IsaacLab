#!/bin/bash

# Conda initialization for current shell
if [ -f "/workspace/miniconda3/bin/conda" ]; then
    eval "$(/workspace/miniconda3/bin/conda shell.bash hook)"
    conda activate env_isaaclab
    echo "Conda environment 'env_isaaclab' activated"
else
    echo "Error: Miniconda not found at /workspace/miniconda3"
fi
