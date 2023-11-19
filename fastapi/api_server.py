#-----------------------------------Libraries-----------------------------------
from fastapi import FastAPI
#from ctransformers import AutoModelForCausalLM
from llama_cpp import Llama  
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pytube import YouTube
from gtts import gTTS
import torch
import os
from dotenv import load_dotenv

#-----------------------------------env variables-----------------------------------

# Load environment variables from .env file
load_dotenv(dotenv_path="../.env")
# Access the variables
url = os.getenv("URL")
MODEL_DIR = os.getenv("MODEL_DIR")
MODEL_PATH = os.getenv("MODEL_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
#print(MODEL_PATH)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
ngpulayers = 20 if torch.cuda.is_available() else 0
#print(device)
#print(MODEL_PATH)
#print(ngpulayers)
# MODEL_DIR = "/app/models"  # Update this to the path inside the container
# MODEL_PATH = os.path.join(MODEL_DIR, "dolphin-2.2.1-mistral-7b.Q6_K.gguf")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
memory = ""
token_count= 0

#-----------------------------------init-----------------------------------
app = FastAPI()

# Cache the ctransformer model
# @app.on_event("startup")
# async def load_model():
#     global llm
#     if not os.path.exists(MODEL_PATH):
#         # Download the model
#         llm = AutoModelForCausalLM.from_pretrained(
#             "TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
#             model_file="dolphin-2.2.1-mistral-7b.Q6_K.gguf",
#             model_type="mistral",
#             gpu_layers=20,
#             max_new_tokens=50,
#             threads=8,
#             temperature=0.2
#         )
#         pass
#     else:
#         print("local files")
#         llm = AutoModelForCausalLM.from_pretrained(
#             MODEL_PATH,
#             local_files_only=True,
#             model_file="dolphin-2.2.1-mistral-7b.Q6_K.gguf",
#             model_type="mistral",
#             gpu_layers=20,
#             max_new_tokens=50,
#             threads=8,
#             temperature=0.2
#         )

#-----------------------------------load model-----------------------------------

@app.on_event("startup")
async def load_model():
    global llm
    if not os.path.exists(MODEL_PATH):
        # Logic for handling the case where the model file doesn't exist
        print("Model file not found!")
        # You might want to download the model file here or raise an exception
    else:
        print("Loading Llama model...")
        llm = Llama(model_path=MODEL_PATH,
                    n_gpu_layers=ngpulayers,
                    n_ctx=1024, n_batch=512, threads=6)
        print("Model loaded successfully.")


if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

#-----------------------------------routes-----------------------------------
# Routes for completion
@app.post("/completion")
async def completion(request: dict):
    global memory
    global token_count
    # user input
    input_text = request["text"]
    # context + user input
    contextual_prompt = memory + "\n" + input_text
    # Define the instruction template
    template = "system\nYou are Dolphin, a helpful AI assistant. Provide the shortest answer possible. Respond to the question to the best of your ability\nuser\n{prompt}\nassistant\n"
    # Format the actual prompt with the template
    formatted_prompt = template.format(prompt=contextual_prompt)
    response = llm(formatted_prompt,
                    max_tokens=80,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=10,
                   )
    text_response = response["choices"][0]["text"]
    token_count += response["usage"]["total_tokens"]
    # Creating the memory string
    memory = f"Prompt: {contextual_prompt}\nResponse: {text_response}"
    #print(response)
    #print(text_response)
    with open(os.path.join(OUTPUT_PATH, "LLM_response.txt"), 'w') as file:
        file.write(memory)
    return {"response": text_response, "token_count": token_count}


@app.post("/download_youtube_audio")
async def download_youtube_audio(url: str):
    YouTube(url).streams.filter(only_audio=True).first().download(filename=os.path.join(OUTPUT_PATH, "output.mp3"))
    return {"info": "Downloaded YouTube audio"}

@app.post("/convert_text_to_speech")
async def convert_text_to_speech(request: dict, language: str = "en"):
    text = request["text"]
    tts = gTTS(text, lang=language, tld="us")
    file_path = os.path.join(OUTPUT_PATH, "speech.mp3")
    tts.save(file_path)
    command = f'mpg321 "{file_path}"'
    os.system(command)
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
    # Save the transcription to output/input.txt
    with open(os.path.join(OUTPUT_PATH, "input.txt"), "w") as file:
        file.write(result_text)
    return result_text

#-----------------------------------main-----------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


#-----------------------------------end-----------------------------------
