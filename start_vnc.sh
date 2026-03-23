#!/bin/bash

echo "=========================================="
echo "Starting VNC Server"
echo "=========================================="
echo ""

# Check if VNC server is already running
if pgrep -x "Xtigervnc" > /dev/null; then
    echo "VNC server is already running on display :1"
    echo "To restart, first run: vncserver -kill :1"
    exit 1
fi

# Start VNC server with optimal settings for Isaac Sim
echo "Starting VNC server on display :1..."
vncserver :1 -geometry 1920x1080 -depth 24 -localhost no

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "VNC Server started successfully!"
    echo "=========================================="
    echo ""
    echo "Connection details:"
    echo "  Display: :1"
    echo "  Port: 5901"
    echo ""
    echo "Make sure to:"
    echo "1. Expose TCP port 5901 in Runpod web interface"
    echo "2. Connect with VNC client to the external address provided"
    echo ""
    echo "To stop VNC server: vncserver -kill :1"
else
    echo ""
    echo "Failed to start VNC server"
    echo "Check if tigervnc is installed: apt-get install -y tigervnc-standalone-server"
fi
