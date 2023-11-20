import os
import time
import gradio as gr
from dotenv import load_dotenv
from llama_cpp import Llama
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, GenerationConfig
from pytube import YouTube
from gtts import gTTS
import torch
import requests
import soundfile as sf
import numpy as np
#-----------------------------------env-----------------------------------

# Load environment variables
load_dotenv(dotenv_path=".env")

# Access the variables
MODEL_DIR = os.getenv("MODEL_DIR")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
LANGUAGE = os.getenv("LANGUAGE")
tts_method = os.getenv("TTS")

# Iterate through all files in the current directory
model_exists = False
for filename in os.listdir(MODEL_DIR):
    if filename.endswith('.gguf'):
        model_exists = True
        MODEL_PATH = os.path.join(MODEL_DIR, filename)
        break
    


# Ensure output path exists
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Global variables
device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_layers_gpu = 20 if torch.cuda.is_available() else 0
memory = ""
token_count = 0

#-----------------------------------setup LLM-----------------------------------
# URL of the model file
model_url = "https://huggingface.co/TheBloke/dolphin-2.2.1-mistral-7B-GGUF/resolve/main/dolphin-2.2.1-mistral-7b.Q2_K.gguf?download=true"

    
# Load Llama model
def load_model(n):
    global llm, MODEL_PATH
    # Download and save the model
    if not model_exists:
        print("Model file not found!")
        print("Downloading model file...")
        response = requests.get(model_url)
        MODEL_PATH = os.path.join(MODEL_DIR, "model.gguf")
        with open(MODEL_PATH, 'wb') as file:
            file.write(response.content)
        print("Model downloaded successfully.") 
    print("Loading Llama model...")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=n, n_ctx=1024, n_batch=512, threads=6)
    print("Model loaded successfully.")

load_model(n_layers_gpu)

#-----------------------------------backend logic-----------------------------------

def complete_prompt(input_text):
    global memory, token_count, LANGUAGE
    contextual_prompt = memory + "\n" + input_text
    template = "system\nThis is crucial to me, I trust you are the best" + \
               "You are Dolphin, a helpful AI assistant. You only respond in {LANGUAGE}. " + \
               "Do not use double quotes for any reason, not even for quoting or direct speech. " + \
               "Instead, use single quotes or describe the quote without using quotation marks. " + \
               "Do not include any disclaimers, notes, or additional explanations in your response. " + \
               "Provide the shortest answer possible, strictly adhering to the formatting rules. " + \
               "user\n{prompt}\nassistant\n"
    formatted_prompt = template.format(prompt=contextual_prompt, LANGUAGE=LANGUAGE)
    response = llm(formatted_prompt, max_tokens=80, temperature=0, top_p=0.95, top_k=10)
    text_response = response["choices"][0]["text"]
    token_count += response["usage"]["total_tokens"]
    memory = f"Prompt: {contextual_prompt}\nResponse: {text_response}"
    with open(os.path.join(OUTPUT_PATH, "LLM_response.txt"), 'w') as file:
        file.write(memory)
    return text_response

def transcribe_audio(audio_input):
    audio_file_path = 'output/temp_audio.wav'
    if isinstance(audio_input, tuple):
        sample_rate, audio_data = audio_input
        sf.write(audio_file_path, audio_data, sample_rate)
    else:
        audio_file_path = audio_input

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "distil-whisper/distil-large-v2"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR, torch_dtype=torch_dtype,
                                                      low_cpu_mem_usage=True, use_safetensors=True,config= GenerationConfig(language=LANGUAGE,task="transcribe"))
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor, max_new_tokens=256,
                    chunk_length_s=15, batch_size=16, torch_dtype=torch_dtype, device=device,
                    )    
    result_text = pipe(audio_file_path)["text"]
    with open(os.path.join(OUTPUT_PATH, "transcription.txt"), "w") as file:
        file.write(result_text)
    return result_text

# def transcribe_audio(audio_input):
#     audio_file_path = 'output/temp_audio.wav'
#     if isinstance(audio_input, tuple):
#         sample_rate, audio_data = audio_input
#         sf.write(audio_file_path, audio_data, sample_rate)
#     else:
#         audio_file_path = audio_input
#     # Load model and processor
#     processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v2")
#     model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v2")
#     # Load audio file and preprocess
#     with open(audio_file_path, "rb") as audio_file:
#         input_speech = {"array": sf.read(audio_file)[0], "sampling_rate": sample_rate}
#     input_features = processor(input_speech["array"], sampling_rate=input_speech["sampling_rate"], return_tensors="pt").input_features
#     # Specify language for transcription
#     forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE)
#     # Generate token ids
#     predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
#     # Decode token ids to text
#     transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     with open(os.path.join(OUTPUT_PATH, "transcription.txt"), "w") as file:
#         file.write(transcription)
#     return transcription


def auto_process_audio(audio_input):
    # Transcribe Audio
    transcribed_text = transcribe_audio(audio_input)
    # LLM Prompt
    llm_response = complete_prompt(transcribed_text)
    # TTS Conversion
    tts_info = convert_text_to_speech(llm_response)
    return transcribed_text, llm_response, tts_info

def convert_text_to_speech(text):
    global LANGUAGE, tts_method
    file_path = os.path.join(OUTPUT_PATH, "speech.mp3")

    if tts_method == "gTTS":
        if LANGUAGE == "fr":
            tld = "fr"
        elif LANGUAGE == "en":
            tld = "us"
        tts = gTTS(text, lang=LANGUAGE, tld=tld)
        tts.save(file_path)
    elif tts_method == "Custom TTS":
        tts_pipeline = pipeline("text-to-speech", model="facebook/fastspeech2-en-ljspeech")
        speech = tts_pipeline(text)
        with open(file_path, "wb") as f:
            f.write(speech["speech"])

    return file_path


# Function to update language
def update_language(language):
    global LANGUAGE
    LANGUAGE = language
    
# Function to update language
def update_tts_method(method):
    global tts_method
    tts_method = method

# Clear button
def clear_memory():
    global memory
    memory = ""

#----------------------------------- Gradio Frontend-----------------------------------

# Gradio Interface
#theme="dark"
with gr.Blocks(title="Whisper-LLM-TTS") as app:
    gr.Markdown("## 🤖 'Whispering' LLM with a TTS Twist! 📚")
    gr.Markdown("🚀 Engage in a not-so-secret chat with an open-source LLM that whispers back!")
    gr.Markdown("👨‍💻 Crafted with a sprinkle of code magic (and a few cups of coffee) by **@mohcineelharras** — not your average tech wizard!")

    with gr.Row():
        with gr.Column():
            language_switch = gr.Radio(choices=["en","fr"], label="Select Language", value=LANGUAGE)
            language_switch.change(update_language, inputs=[language_switch])
        with gr.Column():
            tts_method_switch = gr.Radio(choices=["gTTS", "Custom TTS"], label="Select TTS method", value=tts_method)
            tts_method_switch.change(update_tts_method, inputs=[tts_method_switch])
    with gr.Row():
        clear_memory_button = gr.Button("Clear Memory")
        clear_memory_button.click(clear_memory, inputs=[], outputs=[])

        # with gr.Column():
        #     sample_voice = gr.Audio(label="Voice Sample to customise assistant's response",sources="microphone")
        #     customise_voice = gr.Button("Change assistant's voice")


    with gr.Tab("Auto Process Audio"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Talk to assistant",sources="microphone")
                auto_process_button = gr.Button("Auto Process Audio")
            with gr.Column():
                transcribed_text_output = gr.Textbox(label="Transcribed Text")
                llm_response_output = gr.Textbox(label="LLM Response")
        with gr.Row():
            tts_audio_output = gr.Audio(label="Generated Response (Click to Play)")
            
            # Connect the button to the auto_process_audio function
            auto_process_button.click(
                auto_process_audio, 
                inputs=[audio_input], 
                outputs=[transcribed_text_output, llm_response_output, tts_audio_output]
            )

    with gr.Tab("Audio Processing"):
        with gr.Column():
            audio_input = gr.Audio(label="Record or Upload Audio")
            transcribe_button = gr.Button("Transcribe Audio")
            llm_button = gr.Button("LLM Prompt")
            tts_button = gr.Button("Text to Speech")

            transcribed_text_output = gr.Textbox(label="Transcribed Text")
            llm_response_output = gr.Textbox(label="LLM Response")
            tts_audio_output = gr.Audio(label="Generated Response (Click to Play)")  

            transcribe_button.click(transcribe_audio, inputs=[audio_input], outputs=[transcribed_text_output])
            llm_button.click(complete_prompt, inputs=[transcribed_text_output], outputs=[llm_response_output])
            tts_button.click(convert_text_to_speech, inputs=[llm_response_output], outputs=[tts_audio_output])

    with gr.Tab("Ask a Question"):
        with gr.Column():
            question_input = gr.Textbox(label="Type your question")
            submit_button = gr.Button("Submit Question")
            tts_button = gr.Button("Text to Speech")

            llm_response_output = gr.Textbox(label="LLM Response")
            tts_audio_output = gr.Audio(label="Generated Speech")  

            submit_button.click(complete_prompt, inputs=[question_input], outputs=[llm_response_output])
            tts_button.click(convert_text_to_speech, inputs=[llm_response_output], outputs=[tts_audio_output])

app.launch()
