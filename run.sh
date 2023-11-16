#!/bin/bash

# Function to gracefully stop the running processes
stop_processes() {
    # Stop FastAPI server (if it's running)
    pkill -f "python fastapi/api_server.py"
    
    # Stop Streamlit app (if it's running)
    pkill -f "streamlit run streamlit_app/app.py"
}

# Trap the interrupt signal (Ctrl+C) and call the stop_processes function
trap 'stop_processes; exit 0' SIGINT

# Install Python dependencies from requirements_merged.txt
#pip install -r requirements_merged.txt

# Install required system packages
apt-get update
apt-get install -y python3-pyaudio python3-dev build-essential mpg321

# Launch the FastAPI server in the background
nohup python fastapi/api_server.py > api_server.log 2>&1 &

# Launch the Streamlit app in the background
nohup streamlit run streamlit_app/app.py > app.log 2>&1 &

# Print a message indicating that the services are running
echo "FastAPI server and Streamlit app are now running."

# Keep the script running in the background
while true; do
    sleep 1
done
