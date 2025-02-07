from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import time
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from starlette.responses import StreamingResponse
import asyncio

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

# Chat endpoint with streaming response
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    full_prompt = SYSTEM_PROMPT + "\n" + user_input
    total_tokens = 0
    start_time = time.time()

    async def generate_response():
        nonlocal total_tokens

        try:
            # Start generating tokens in streaming mode.
            # Note: Ensure your vLLM.generate supports the streaming parameter.
            token_stream = llm.generate([full_prompt], sampling_params, streaming=True)
            
            # Iterate over tokens as they are generated.
            for output in token_stream:
                # Depending on vLLM's API, the token may be available as output.token or output.text.
                token = getattr(output, "token", output.text)
                yield token + " "
                total_tokens += 1
                await asyncio.sleep(0)  # Yield control to the event loop

            # Append metadata after generation finishes
            generation_time = time.time() - start_time
            tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0
            yield f"\n\n[TOKENS USED: {total_tokens}, TPS: {tokens_per_second:.2f}, TIME: {generation_time:.2f}s]"
        
        except asyncio.CancelledError:
            # Client disconnected, exit gracefully
            print("Client disconnected. Stopping response stream.")
            return

    return StreamingResponse(generate_response(), media_type="text/plain")

# Run FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
