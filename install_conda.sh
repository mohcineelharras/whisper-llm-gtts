#!/bin/bash

#---------------------- Adapt this if you run it on windows -----------------------

# Check if conda is installed by trying to locate the 'conda' command
if ! command -v conda &> /dev/null
then
    echo "Conda not found, installing Miniforge..."

    # Download and install Miniforge (or Mambaforge if preferred)
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3

    # Initialize conda for the current shell
    source $HOME/miniforge3/etc/profile.d/conda.sh

    # Uncomment the line below if needed
    conda init
    echo "Great ! conda is now installed"

    # Clean up the installer
    rm Miniforge3-Linux-x86_64.sh
else
    echo "Conda is already installed"
fi

# Set the environment variable for CMAKE_ARGS
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

# Install required system packages
sudo apt-get update
sudo apt-get install -y python3-pyaudio python3-dev build-essential mpg321

# Create env
conda create -n "audio" python=3.11 -y

# Activate env
source activate audio

# Install cudatoolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# Install llama-cpp-python with specific CMAKE_ARGS
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
pip install -r requirements_merged.txt
