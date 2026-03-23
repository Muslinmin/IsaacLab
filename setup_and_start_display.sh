
#!/bin/bash

echo "=========================================="

echo "Master Setup Script"

echo "=========================================="

echo ""

# Step 1: Run the installation script

if [ -f "/workspace/setup_runpod_vnc.sh" ]; then

    echo "Running VNC installation script..."

    bash /workspace/setup_runpod_vnc.sh

    

    if [ $? -ne 0 ]; then

        echo "Error: Installation script failed"

        exit 1

    fi

else

    echo "Error: setup_runpod_vnc.sh not found in /workspace"

    exit 1

fi

echo ""

echo "Waiting 3 seconds before starting VNC server..."

sleep 3

echo ""

# Step 2: Start VNC server

if [ -f "/workspace/start_vnc.sh" ]; then

    echo "Starting VNC server..."

    bash /workspace/start_vnc.sh

    

    if [ $? -ne 0 ]; then

        echo "Error: VNC server failed to start"

        exit 1

    fi

else

    echo "Error: start_vnc.sh not found in /workspace"

    exit 1

fi

echo ""

echo "=========================================="

echo "Setup and VNC server launch complete!"

echo "=========================================="

echo ""

echo "Your VNC server is now running!"

echo "Remember to expose TCP port 5901 in Runpod and connect with your VNC client."

