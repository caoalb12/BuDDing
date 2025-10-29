#!/bin/bash

set -Eeuo pipefail

echo "Step 1: Updating apt packages"
sudo apt update -y
sudo apt install -y python3-pip python3.10-venv

cd /local/repository

echo "Step 2: Activating virtual environment"
python3 -m venv env
source env/bin/activate

echo "Step 3: Installing torch and transformers"
pip install --upgrade pip
pip install torch
pip install transformers

# echo "Step 1: Updating apt packages"
# sudo apt update -y

# echo "Step 2: Installing pip"
# sudo curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
# sudo python3 /tmp/get-pip.py

# echo "Step 3: Installing PyTorch and NumPy"
# sudo pip3 install --upgrade pip
# sudo pip3 install torch numpy transformers

# echo "Step 4: Installing NVIDIA utilities and driver"
# sudo apt install -y nvidia-utils-580 nvidia-driver-580

# sudo rm -f /tmp/get-pip.py

# echo "Step 5: Rebooting system"
# sudo reboot
