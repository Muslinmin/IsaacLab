#!/bin/bash

# Initialize conda
if [ -f "/workspace/miniconda3/bin/conda" ]; then
    /workspace/miniconda3/bin/conda init bash
    echo "Conda initialized in ~/.bashrc"
    echo "Run 'source ~/.bashrc' to activate conda"
else
    echo "Error: Miniconda not found at /workspace/miniconda3"
fi