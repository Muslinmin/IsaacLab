#!/bin/bash

echo "=========================================="
echo "Runpod VNC + Isaac Sim Setup Script"
echo "=========================================="
echo ""
# Update package list
echo "Updating package list..."
apt-get update -qq

# Install VNC and desktop environment
echo "Installing VNC server and XFCE desktop..."
apt-get install -y xfce4 xfce4-goodies tigervnc-standalone-server tigervnc-common

# Install terminal emulator
echo "Installing terminal emulator..."
apt-get install -y xfce4-terminal

# Install browser (optional)
echo "Installing Chromium browser..."
apt-get install -y chromium-browser

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start VNC server: vncserver :1 -geometry 1920x1080 -depth 24 -localhost no"
echo "2. In Runpod web interface, expose TCP port 5901"
echo "3. Connect with VNC client to the provided external address"
echo "4. Download Isaac Sim if not already done"
echo ""
