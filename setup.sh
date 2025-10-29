#!/bin/bash

set -euo pipefail

echo "Step 1: Updating apt packages"
sudo apt update -y

echo "Step 2: Installing pip"
sudo curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
sudo python3 /tmp/get-pip.py

echo "Step 3: Installing PyTorch and NumPy"
sudo pip3 install --upgrade pip
sudo pip3 install torch numpy transformers

echo "Step 4: Installing NVIDIA utilities and driver"
sudo apt install -y nvidia-utils-580 nvidia-driver-580

sudo rm -f /tmp/get-pip.py

# echo "Step 5: Rebooting system"
# sudo reboot