from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.responses import StreamingResponse, Response

# Initialize FastAPI app
app = FastAPI()

# Static files and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Model path
MODEL_PATH = "deepseek-r1-awq"

# Load the model
llm = LLM(
    model=MODEL_PATH,
    gpu_memory_utilization=0.98,
    max_model_len=8192
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7, 
    top_p=0.9, 
    max_tokens=8192
)

# System prompt
SYSTEM_PROMPT = """
Think step by step but be concise. Prioritize complete answers over verbose reasoning.
Generate long, exhaustive outputs where needed while maintaining clarity.
"""

# Pydantic model for chat request
class ChatRequest(BaseModel):
    user_input: str

# Root endpoint
@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chat endpoint with correct streaming response
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    full_prompt = SYSTEM_PROMPT + "\n" + user_input
    total_tokens = 0
    start_time = time.time()

    def generate_response():
        nonlocal total_tokens
        prompt = full_prompt

        for _ in range(3):  # Limit to 3 rounds
            output = llm.generate([prompt], sampling_params)
            chunk = output[0].outputs[0].text.strip()
            
            if not chunk:
                break

            # Yield each token in the chunk
            for token in chunk.split():
                yield token + " "
                total_tokens += 1

            if len(chunk) < sampling_params.max_tokens or "### END" in chunk:
                break  
            
            prompt += " " + chunk  

    # Stream response and send headers after generation
    return StreamingResponse(generate_response(), media_type="text/plain")

# Run FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)