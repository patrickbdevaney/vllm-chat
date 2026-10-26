"""In-process chat server: vLLM loaded inside FastAPI, full (non-streaming) responses.

Run:   python app/server.py            ->  http://localhost:8000
Config (environment):
  VLLM_MODEL               HF id or local path of an AWQ DeepSeek-R1-Distill checkpoint
  MAX_MODEL_LEN            context length (default 8192; 4096 is faster to prefill)
  GPU_MEMORY_UTILIZATION   fraction of VRAM vLLM may claim (default 0.98)
  HOST, PORT               bind address (default 127.0.0.1:8000)
"""
import os
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from vllm import LLM, SamplingParams

MODEL = os.environ.get("VLLM_MODEL", "casperhansen/deepseek-r1-distill-qwen-7b-awq")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.98"))

app = FastAPI()
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Loaded once at import; every request reuses the same engine.
llm = LLM(
    model=MODEL,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=MAX_MODEL_LEN,
)

SYSTEM_PROMPT = """
Think step by step but be concise. Prioritize complete answers over verbose reasoning.
Generate long, exhaustive outputs where needed while maintaining clarity.
"""


class ChatRequest(BaseModel):
    user_input: str


@app.get("/")
async def get_index(request: Request):
    return templates.TemplateResponse(request, "index.html")


# Returns the whole response at once, with token metrics.
@app.post("/chat")
async def chat(request: ChatRequest):
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="No input provided")

    full_prompt = SYSTEM_PROMPT + "\n" + user_input
    total_tokens = 0
    start_time = time.time()
    response_text = ""

    for _ in range(3):  # Continue a truncated answer for at most 3 rounds
        output = llm.generate([full_prompt], sampling_params)
        chunk = output[0].outputs[0].text.strip()

        if not chunk:
            break

        response_text += chunk + "\n"
        total_tokens += len(output[0].outputs[0].token_ids)

        if len(chunk) < sampling_params.max_tokens or "### END" in chunk:
            break

        full_prompt += " " + chunk  # Continue with previous chunk

    generation_time = time.time() - start_time
    tokens_per_second = total_tokens / generation_time if generation_time > 0 else 0

    return JSONResponse(content={
        "response": response_text,
        "tokens": total_tokens,
        "tokens_per_second": tokens_per_second,
        "generation_time": generation_time,
    })


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
