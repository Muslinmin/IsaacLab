#!/bin/bash

# Isaac Sim Environment Setup
export ISAACSIM_PATH=/workspace/isaacsim
export DISPLAY=:1
export OMNI_KIT_ALLOW_ROOT=1
export PATH=$ISAACSIM_PATH:$PATH

echo "Isaac Sim environment configured:"
echo "  ISAACSIM_PATH = $ISAACSIM_PATH"
echo "  DISPLAY = $DISPLAY"
echo "  OMNI_KIT_ALLOW_ROOT = $OMNI_KIT_ALLOW_ROOT"