#-----------------------------------Libraries-----------------------------------
from fastapi import FastAPI
from fastapi import UploadFile, File
from ctransformers import AutoModelForCausalLM
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube
from gtts import gTTS
import torch
import requests
import os
#-----------------------------------env variables-----------------------------------

MODEL_DIR = "../models"
MODEL_NAME = "TheBloke/dolphin-2.2.1-mistral-7B-GGUF"
MODEL_PATH = os.path.join(MODEL_DIR, "dolphin-2.2.1-mistral-7b.Q6_K.gguf")
device = "cuda:0" if torch.cuda.is_available() else "cpu"



#-----------------------------------init-----------------------------------
app = FastAPI()
# Cache the ctransformer model
@app.on_event("startup")
async def load_model():
    global ctransformer_model
    if not os.path.exists(MODEL_PATH):
        # Download the model
        ctransformer_model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
            model_file="dolphin-2.2.1-mistral-7b.Q6_K.gguf",
            model_type="mistral",
            gpu_layers=20,
            max_new_tokens=50,
            threads=8,
            temperature=0.2
        )
        pass
    else:
        print("local files")
        ctransformer_model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            model_file="dolphin-2.2.1-mistral-7b.Q6_K.gguf",
            model_type="mistral",
            gpu_layers=20,
            max_new_tokens=50,
            threads=8,
            temperature=0.2
        )


if not os.path.exists("../output"):
    os.makedirs("../output")

#-----------------------------------routes-----------------------------------
# Routes for completion
@app.post("/completion")
async def completion(request: dict):
    input_text = request["text"]
    # Define the instruction template
    template = "system\nYou are Dolphin, a helpful AI assistant. Provide the shortest answer possible. Respond to the question to the best of your ability\nuser\n{prompt}\nassistant\n"
    # Format the actual prompt with the template
    formatted_prompt = template.format(prompt=input_text)
    response = ctransformer_model(formatted_prompt)
    with open('../output/LLM_response.txt', 'w') as file:
        file.write(response)
    return {"response": response}


@app.post("/download_youtube_audio")
async def download_youtube_audio(url: str):
    YouTube(url).streams.filter(only_audio=True).first().download(filename="../output/output.mp3")
    return {"info": "Downloaded YouTube audio"}

@app.post("/convert_text_to_speech")
async def convert_text_to_speech(request: dict, language: str = "en"):
    text = request["text"]
    tts = gTTS(text, lang=language, tld="us")
    tts.save("../output/speech.mp3")
    os.system("mpg321 ../output/speech.mp3")
    return {"info": "Converted text to speech"}



@app.post("/transcribe_audio")
async def transcribe_audio(file_path: str):
    # Set up Whisper model
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
    result_text = pipe(file_path)["text"]
    # Save the transcription to ../output/input.txt
    with open("../output/input.txt", "w") as file:
        file.write(result_text)
    return result_text

#-----------------------------------main-----------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#-----------------------------------end-----------------------------------
