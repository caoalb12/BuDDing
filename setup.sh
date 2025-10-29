#!/bin/bash

set -Eeuo pipefail

echo "Step 1: Updating apt packages"
sudo apt update -y
sudo apt install -y python3-pip python3.10-venv
sudo apt install -y nvidia-driver-550
sudo modprobe nvidia
sudo modprobe nvidia_uvm

echo "NVIDIA DEBUGGING PURPOSES ONLY:"
lsmod | grep nvidia
echo "-------------------------------"

nvidia-smi

cd /local/repository

echo "Step 2: Activating virtual environment"
python3 -m venv env
source env/bin/activate

echo "Step 3: Installing torch and transformers"
pip install --upgrade pip
pip install torch
pip install transformers
