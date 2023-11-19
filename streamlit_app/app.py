#-----------------------------------Libraries-----------------------------------
import streamlit as st
import os
from dotenv import load_dotenv
import time 
import requests
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from record_audio import record_audio

#-----------------------------------env-----------------------------------

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")

# Access the variables
url = os.getenv("URL")
MODEL_DIR = os.getenv("MODEL_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
LANGUAGE = os.getenv("LANGUAGE")

#print(OUTPUT_PATH)
#-----------------------------------fastapi functions-----------------------------------

# FastAPI request function
# Example of sending a file

def download_youtube_audio(video_url):
    response = requests.post(
        f"http://{url}:8000/download_youtube_audio",
        json={"text": video_url}  # Send data as JSON
    )
    if response.status_code == 200:
        return response.json()['response']
    else:
        return "Error: " + response.text

def convert_text_to_speech(text, language=LANGUAGE):
    response = requests.post(
        f"http://{url}:8000/convert_text_to_speech",
        json={"text": text}  # Send data as JSON
    )
    if response.status_code == 200:
        return response.json()['info']
    else:
        return "Error: " + response.text

def get_response_from_hf_transformers(text):
    response = requests.post(
        f"http://{url}:8000/completion",
        json={"text": text}  # Send data as JSON
    )
    if response.status_code == 200:
        return response.json()['response'], response.json()['token_count']
    else:
        return "Error: " + response.text

# Updated transcribe_audio function to save transcription
def transcribe_audio(audio_file_path):
    response = requests.post(
        f"http://{url}:8000/transcribe_audio",
        params={"file_path": audio_file_path}  # Include the file_path parameter
    )
    if response.status_code == 200:
        #transcription_result = response.json()['response']
        return response.json()
    else:
        return "Error: " + response.text
    
#-----------------------------------streamlit frontend-----------------------------------

#App
def main():
    with st.sidebar:
        st.title("🤖 whisper - LLM - TTS 📚")
        st.write("🚀 Talk to an open source LLM!")
        st.write("This app is developed and maintained by **@mohcineelharras**")
            
    st.title('Alexa like assistant')

    # Create Tabs
    #tab1, tab2, tab3, tab4 = st.tabs(["Record Audio", "Upload Audio File", "Ask a Question", "YouTube URL Processing"])
    tab1, tab2, tab3 = st.tabs(["Record Audio", "Upload Audio File", "Ask a Question"])
    # Record Audio Tab (Placeholder)
    with tab1:
        audio_file_path = os.path.join(OUTPUT_PATH, "recording.wav")
        st.header("Record Audio")
        if st.button('Record Audio'):
            record_audio()
            st.write(f"Recording complete. File saved as {audio_file_path}")

            # Process the recorded audio file
            start_time = time.time()
            st.header("Transcription")
            with st.spinner('Transcribing...'):
                response = requests.post(
                    "http://localhost:8000/transcribe_audio",  # Replace with your actual endpoint URL
                    json={"file_path": audio_file_path}  # Send data as JSON
                )

                if response.status_code == 200:
                    transcribed_text = response.json()
                else:
                    transcribed_text = f"Error: {response.text}"
            elapsed_time = time.time() - start_time
            st.write(f"Transcribed in {elapsed_time:.2f} seconds: ")
            st.write("Transcription :",transcribed_text)

            # Send to LLM and process response
            start_time = time.time()
            st.header("LLM prompting")
            with st.spinner('Processing with LLM...'):
                #llm_response = get_response_from_gpt(transcribed_text)
                llm_response, token_count = get_response_from_hf_transformers(transcribed_text)
            elapsed_time = time.time() - start_time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)
            st.write("Token count so far :",token_count)

            # Text to Speech
            start_time = time.time()
            st.header("Text to speech")
            convert_text_to_speech(llm_response, language="en")
            elapsed_time = time.time() - start_time
            st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
            audio_file = os.path.join(OUTPUT_PATH, "speech.mp3")
            if os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3', start_time=0)
            else:
                st.error("Audio file not found.")

    # Upload Audio File Tab
    with tab2:
        uploaded_file = st.file_uploader("Upload an audio file:", type=['wav', 'mp3', 'm4a'])
        if uploaded_file is not None:
            audio_file_path = os.path.join(OUTPUT_PATH, uploaded_file.name)
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("Uploaded audio file.")
            # Automatically start transcribing after file upload
            start_time = time.time()
            st.header("Transcription")
            with st.spinner('Transcribing...'):
                response = requests.post(
                    "http://localhost:8000/transcribe_audio",  # Replace with your actual endpoint URL
                    json={"file_path": audio_file_path}  # Send data as JSON
                )

                if response.status_code == 200:
                    transcribed_text = response.json()
                else:
                    transcribed_text = f"Error: {response.text}"
            elapsed_time = time.time() - start_time
            st.write(f"Transcribed in {elapsed_time:.2f} seconds: ")
            st.write("Transcription :",transcribed_text)

            # Automatically send transcription to LLM
            start_time = time.time()
            st.header("LLM prompting")
            with st.spinner('Processing with LLM...'):
                #llm_response = get_response_from_gpt(transcribed_text)
                llm_response, token_count = get_response_from_hf_transformers(transcribed_text)
            elapsed_time = time.time() - start_time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)
            st.write("Token count so far :",token_count)

            # Automatically generate text to speech
            start_time = time.time()
            st.header("Text2Speech")
            convert_text_to_speech(llm_response, language="en")
            elapsed_time = time.time() - start_time
            st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
            audio_file = os.path.join(OUTPUT_PATH, "speech.mp3")
            if os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3', start_time=0)
            else:
                st.error("Audio file not found.")
            

    # Ask a Question Tab
    with tab3:
        user_question = st.text_input('Type your question here:', key="question_input")

        if user_question:
            start_time = time.time()  # Start the timer
            st.header("LLM prompting")
            with st.spinner('Processing...'):
                #llm_response = get_response_from_gpt(user_question)
                llm_response, token_count = get_response_from_hf_transformers(user_question)
            elapsed_time = time.time() - start_time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)
            st.write("Token count so far :",token_count)

            if llm_response:
                st.header("Text2Speech")
                start_time = time.time()  # Start the timer for TTS
                convert_text_to_speech(llm_response, language="en")
                elapsed_time = time.time() - start_time  # Calculate elapsed time for TTS
                st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
                audio_file = os.path.join(OUTPUT_PATH, "speech.mp3")
                if os.path.exists(audio_file):
                    st.audio(audio_file, format='audio/mp3', start_time=0)
                else:
                    st.error("Audio file not found.")

if __name__ == "__main__":
    main()


#-----------------------------------end-----------------------------------
