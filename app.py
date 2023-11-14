import streamlit as st
from pytube import YouTube
import torch
#from TTS.api import TTS
from gtts import gTTS
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from langchain import llms
import time  # Import time module
import sounddevice as sd
import soundfile as sf
import numpy as np


# Set up OpenAI and TTS
api_base = "http://127.0.0.1:5001/v1" # point to the local server
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
#tts.to(device=device)

def save_uploadedfile(uploadedfile):
    with open(os.path.join("output", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to output".format(uploadedfile.name))

def download_youtube_audio(video_url):
    YouTube(video_url).streams.filter(only_audio=True).first().download(filename="output/output.mp3")

def convert_text_to_speech(text, language="en"):
    tts = gTTS(text, lang=language, tld="us")
    tts.save("output/speech.mp3")
    os.system("mpg321 output/speech.mp3")  # or use another player compatible with your system

def get_response_from_gpt(text):
    # Define the instruction template
    template = "system\nYou are Dolphin, a helpful AI assistant. Provide the shortest answer possible. Respond to the question to the best of your ability\nuser\n{prompt}\nassistant\n"
    # Format the actual prompt with the template
    formatted_prompt = template.format(prompt=text)
    # Initialize the LLM with the required configuration
    llm = llms.OpenAI(
        openai_api_key="anyValueYouLike",
        temperature=0.2,
        openai_api_base=api_base,
        max_tokens=100,
        # Add other relevant configurations like cpu, threads, etc.
    )
    # Get the response using the formatted prompt
    response = llm(formatted_prompt)
    return response


# Updated transcribe_audio function to save transcription
def transcribe_audio(audio_file_path):
    # Set up Whisper model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v2"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=256,
        chunk_length_s=15,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )
    result_text = pipe(audio_file_path)["text"]
    # Save the transcription to output/input.txt
    with open("output/input.txt", "w") as file:
        file.write(result_text)
    return result_text
import subprocess

def record_audio(duration=5, fs=44100, filename="output/recording.wav"):
    try:
        print("Recording...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        print("Recording finished, saving file...")
        sf.write(filename, audio_data, fs)
        print(f"File saved as {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if not os.path.exists("output"):
        os.makedirs("output")
        
    st.title('mohcineelharras AI assistant')

    # Create Tabs
    #tab1, tab2, tab3, tab4 = st.tabs(["Record Audio", "Upload Audio File", "Ask a Question", "YouTube URL Processing"])
    tab1, tab2, tab3 = st.tabs(["Record Audio", "Upload Audio File", "Ask a Question"])
    # Record Audio Tab (Placeholder)
    with tab1:
        audio_file_path = 'output/recording.wav'
        st.header("Record Audio")
        if st.button('Record Audio'):
            record_audio()
            st.write(f"Recording complete. File saved as {audio_file_path}")

            # Process the recorded audio file
            start_time = time.time()
            st.header("Transcription")
            with st.spinner('Transcribing...'):
                transcribed_text = transcribe_audio(audio_file_path)
            elapsed_time = time.time() - start_time
            st.write(f"Transcribed in {elapsed_time:.2f} seconds: ")
            st.write("Transcription :",transcribed_text)

            # Send to LLM and process response
            start_time = time.time()
            st.header("LLM prompting")
            with st.spinner('Processing with LLM...'):
                llm_response = get_response_from_gpt(transcribed_text)
                with open('output/LLM_response.txt', 'w') as file:
                    file.write(llm_response)
            elapsed_time = time.time() - start_time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)

            # Text to Speech
            start_time = time.time()
            convert_text_to_speech(llm_response, language="en")
            elapsed_time = time.time() - start_time
            st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
            audio_file = 'output/speech.mp3'
            if os.path.exists(audio_file):
                st.audio(audio_file, format='audio/mp3', start_time=0)
            else:
                st.error("Audio file not found.")

    # Upload Audio File Tab
    with tab2:
        uploaded_file = st.file_uploader("Upload an audio file:", type=['wav', 'mp3', 'm4a'])
        if uploaded_file is not None:
            audio_file_path = os.path.join("output", uploaded_file.name)
            with open(audio_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("Uploaded audio file.")

            # Automatically start transcribing after file upload
            start_time = time.time()
            st.header("Transcription")
            with st.spinner('Transcribing...'):
                transcribed_text = transcribe_audio(audio_file_path)
            elapsed_time = time.time() - start_time
            st.write(f"Transcribed in {elapsed_time:.2f} seconds: ")
            st.write("Transcription :",transcribed_text)

            # Automatically send transcription to LLM
            start_time = time.time()
            st.header("LLM prompting")
            with st.spinner('Processing with LLM...'):
                llm_response = get_response_from_gpt(transcribed_text)
                with open('output/LLM_response.txt', 'w') as file:
                    file.write(llm_response)
            elapsed_time = time.time() - start_time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)

            # Automatically generate text to speech
            st.write("Generating text to speech: ")
            start_time = time.time()
            st.header("Text2Speech")
            convert_text_to_speech(llm_response, language="en")
            elapsed_time = time.time() - start_time
            st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
            audio_file = 'output/speech.mp3'
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
                llm_response = get_response_from_gpt(user_question)
                with open('output/LLM_response.txt', 'w') as file:
                    file.write(llm_response)
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            st.write(f"LLM response in {elapsed_time:.2f} seconds: ")
            st.write("LLM response :",llm_response)

            if llm_response:
                st.header("Text2Speech")
                start_time = time.time()  # Start the timer for TTS
                convert_text_to_speech(llm_response, language="en")
                elapsed_time = time.time() - start_time  # Calculate elapsed time for TTS
                st.write(f"Text to speech generated in {elapsed_time:.2f} seconds.")
                audio_file = 'output/speech.mp3'
                if os.path.exists(audio_file):
                    st.audio(audio_file, format='audio/mp3', start_time=0)
                else:
                    st.error("Audio file not found.")

    # YouTube URL Processing Tab
    # with tab4:
    #     youtube_url = st.text_input('Enter YouTube URL:')
    #     if youtube_url and st.button('Process YouTube Audio'):
    #         with st.spinner('Downloading and processing audio...'):
    #             download_youtube_audio(youtube_url)
    #             st.success("Downloaded and processed YouTube audio.")
    #             # Add processing logic if needed

if __name__ == "__main__":
    main()
