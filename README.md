# Setup Instructions

1. SSH into the remote machine.
2. Clone BuDDing repository.
2. Run `chmod +x ./setup.sh`.
3. Run `./setup.sh`. This will install the necessary libraries and then reboot the machine.
4. Monitor the CloudLab UI. When the node is ready again, SSH back into the machine.
5. Set your HuggingFace token with `export HF_TOKEN=...`.

# CloudLab Experiment Instructions

Before starting experiment if not using hardware that’s NOT r7525:

1. Go to BuDDing GitHub -> profile.py and change node.hardware_type variable to be the hardware you’re using (do not need to specify region it should automatically work)
2. Text me to update the profile on cloudlab (😭)

When starting experiment:

1. Start experiment like normal
2. When you get to the point where it says the experiment is "Finished" check the service logs
	a. click the gear in list view
	b. click on Execute Service Logs
3. Cmd F to find "nvidia-smi"
	a. if there isn't any output for nvidia-smi, then go to the gear -> reboot node and wait
	b. if the box of GPU info shows up then you're good to go
4. For some reason rename profile.py to temp.py (it's too hard to explain just do it)
    a. DO NOT push temp.py to github
5. Set HF_TOKEN or smth idk
