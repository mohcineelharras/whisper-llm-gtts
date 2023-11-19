#!/bin/bash

#---------------------- Adapt this if you run it on windows -----------------------


# Set the environment variable for CMAKE_ARGS
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

# Install required system packages
sudo apt-get update
sudo apt-get install -y python3-pyaudio python3-dev build-essential mpg321


# Install torch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# cudatoolkit
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# Install llama-cpp-python with specific CMAKE_ARGS
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
pip install -r requirements_merged.txt
