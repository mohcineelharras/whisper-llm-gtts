#!/bin/bash

#---------------------- Adapt this if you run it on windows -----------------------

# Default paths where Conda might be installed
CONDA_PATHS=("$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "$HOME/miniforge3/bin/conda")

CONDA_INSTALLED=false
for conda_path in "${CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        CONDA_INSTALLED=true
        echo "Conda found at $conda_path"
        break
    fi
done

if [ "$CONDA_INSTALLED" = false ]; then
    echo "Conda not found, installing Miniforge..."

    # Download and install Miniforge (or Mambaforge if preferred)
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda

    # Initialize conda for the current shell
    source /opt/conda/etc/profile.d/conda.sh

    # Uncomment the line below if needed
    conda init bash
    echo "Great ! conda is now installed"

    # Clean up the installer
    rm Miniforge3-Linux-x86_64.sh
else
    echo "Conda is already installed"
fi

# Set the environment variable for CMAKE_ARGS
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

# Install required system packages
apt-get update
apt-get install -y python3-pyaudio python3-dev build-essential mpg321

# Create env
conda create -n "fastapi-server" python=3.11 -y

# Activate env
#source activate fastapi-server

# ffmpeg
conda run -n fastapi-server pip install ffmpeg-python -y

# Install torch correctly
conda run -n fastapi-server pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install cudatoolkit
conda run -n fastapi-server conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit -y

# Install llama-cpp-python with specific CMAKE_ARGS
env CMAKE_ARGS="-DLLAMA_CUBLAS=on" conda run -n fastapi-server pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Install other dependencies
conda run -n fastapi-server pip install -r requirements.txt

