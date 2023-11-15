from fastapi import FastAPI
from ctransformers import AutoModelForCausalLM
from pydantic import BaseModel

class CompletionRequest(BaseModel):
    text: str

app = FastAPI()

# Cache the ctransformer model
@app.on_event("startup")
async def load_model():
    global ctransformer_model
    ctransformer_model = AutoModelForCausalLM.from_pretrained(
        "TheBloke/dolphin-2.2.1-mistral-7B-GGUF",
        model_file="dolphin-2.2.1-mistral-7b.Q4_K_M.gguf",
        model_type="mistral",
        gpu_layers=20,
        max_new_tokens=50,
        threads=8,
        temperature=0.2
    )

# Route for completion
@app.post("/completion")
async def completion(request: CompletionRequest):
    input_text = request.text
    # Define the instruction template
    template = "system\nYou are Dolphin, a helpful AI assistant. Provide the shortest answer possible. Respond to the question to the best of your ability\nuser\n{prompt}\nassistant\n"
    # Format the actual prompt with the template
    formatted_prompt = template.format(prompt=input_text)

    response = ctransformer_model(formatted_prompt)
    return {"response": response}
