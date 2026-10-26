from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time

# Initialize FastAPI app
app = FastAPI()

# Model name (already in AWQ format, no need for quantization)
MODEL_NAME = "inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ"

# Load the model with optimized settings (load globally to avoid reloading on every request)
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.98,  # Use almost all available VRAM
    max_model_len=4096  # Reduce to fit within GPU memory
)

# Sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)

# Pydantic model for chat request
class ChatRequest(BaseModel):
    user_input: str

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    # Measure start time
    start_time = time.time()

    # Generate response
    outputs = llm.generate([user_input], sampling_params)
    model_response = outputs[0].outputs[0].text

    # Performance metrics
    generation_time = time.time() - start_time
    num_tokens = len(outputs[0].outputs[0].token_ids)
    tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0

    return {
        "response": model_response,
        "generation_time": f"{generation_time:.2f}s",
        "tokens_per_second": f"{tokens_per_second:.2f} tok/sec",
        "tokens": num_tokens
    }