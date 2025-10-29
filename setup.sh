#!/bin/bash

set -euo pipefail

echo "Step 1: Updating apt packages"
apt update -y

echo "Step 2: Installing pip"
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
python3 /tmp/get-pip.py

echo "Step 3: Installing PyTorch and NumPy"
pip3 install --upgrade pip
pip3 install torch numpy transformers

echo "Step 4: Installing NVIDIA utilities and driver"
apt install -y nvidia-utils-580 nvidia-driver-580

rm -f /tmp/get-pip.py

echo "Step 5: Rebooting system"
reboot