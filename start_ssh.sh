#!/bin/bash
mkdir -p /root/.ssh
cp /workspace/.ssh/id_ed25519 /root/.ssh/id_ed25519
chmod 600 /root/.ssh/id_ed25519
eval "$(ssh-agent -s)"
ssh-add /root/.ssh/id_ed25519
git config --global user.email "crusader2adventureabjure@gmail.com"
git config --global user.name "lalawee"
