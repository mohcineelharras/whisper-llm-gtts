#!/bin/bash

# Sourcing env variables
if [ -f "./.env" ]; then
    export $(grep -v '^#' ./.env | xargs)
fi
#export CUDA_VISIBLE_DEVICES=0
#export FORCE_CMAKE=1
#export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
#export URL=localhost
#OUTPUT_PATH="output"
#export MODEL_PATH=../models/dolphin-2.2.1-mistral-7b.Q6_K.gguf

# Function to record audio using Python (similar to Streamlit app)
record_audio() {
    local output_path=$1
    local duration=$2
    conda run -n audio python streamlit_app/utils/record_audio.py "$output_path" $duration
}


# Function to start the FastAPI server
start_server() {
    python api_server.py &
    SERVER_PID=$!
    echo "FastAPI server started with PID $SERVER_PID"
}

# Function to clean up and exit
cleanup() {
    echo "Stopping FastAPI server..."
    kill $SERVER_PID
    echo "FastAPI server stopped."
    exit
}

# Check for -y argument for scheduled recording
SCHEDULED=false
RECORD_DURATION=5

if [ "$1" == "-y" ]; then
    SCHEDULED=true
    RECORD_DURATION=5 
fi

# Start the FastAPI server
cd fastapi
echo "Launching server .."
start_server
cd ../

# Trap script termination to cleanup
trap cleanup INT TERM

while true; do
    if [ "$SCHEDULED" != true ]; then
        sleep 3
        read -p "Press Enter to start recording for $RECORD_DURATION seconds..." 
    else
        sleep 3
        echo "Waiting for $RECORD_DURATION seconds to record..."
        sleep $RECORD_DURATION
    fi

    # Record audio
    record_audio "output/recording.wav" $RECORD_DURATION

    # Define the audio file path
    AUDIO_FILE="$OUTPUT_PATH/recording.wav"
    echo -e "\n"
    echo $AUDIO_FILE
    echo "Sending audio for transcription..."
    # Send audio file path to FastAPI server for transcription
    TRANSCRIPTION_RESPONSE=$(curl -X POST -H "Content-Type: application/json" -d "{\"file_path\": \"$AUDIO_FILE\"}" http://localhost:8000/transcribe_audio)

    # Check if transcription was successful
    if [ -z "$TRANSCRIPTION_RESPONSE" ] || [ "$TRANSCRIPTION_RESPONSE" == "null" ]; then
        echo "Transcription failed."
        continue
    else
        echo "Transcription successful. Text: $TRANSCRIPTION_RESPONSE"
    fi

    echo -e "\n"
    echo "Sending transcription to LLM..."
    echo "{\"text\": $TRANSCRIPTION_RESPONSE}"
    # Send transcription to LLM
    LLM_RESPONSE=$(curl -X POST -H "Content-Type: application/json" -d "{\"text\": $TRANSCRIPTION_RESPONSE}" http://localhost:8000/completion)
    LLM_TEXT=$(echo $LLM_RESPONSE | jq -r '.response') # Extract the LLM response

    # Check if LLM response was successful
    if [ -z "$LLM_TEXT" ] || [ "$LLM_TEXT" == "null" ]; then
        echo "LLM processing failed."
        continue
    else
        echo "LLM response successful. Text: $LLM_TEXT"
    fi

    echo -e "\n"
    echo "Sending text for speech conversion..."
    echo "{\"text\": \"$LLM_TEXT\", \"language\": \"$LANGUAGE\"}"
    TTS_RESPONSE=$(curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"$LLM_TEXT\", \"language\": \"$LANGUAGE\"}" http://localhost:8000/convert_text_to_speech)

    # Check if TTS was successful
    if [ "$TTS_RESPONSE" != "{\"info\":\"Converted text to speech\"}" ]; then
        echo "Text-to-Speech conversion failed."
        continue
    else
        echo "Text-to-Speech conversion successful."
    fi

done


