# Setup Instructions

1. SSH into the remote machine.
2. Clone BuDDing repository.
2. Run `chmod +x ./setup.sh`.
3. Run `./setup.sh`. This will install the necessary libraries and then reboot the machine.
4. Monitor the CloudLab UI. When the node is ready again, SSH back into the machine.
5. Set your HuggingFace token with `export HF_TOKEN=...`.