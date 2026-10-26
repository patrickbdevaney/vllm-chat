import asyncio
import json
import ssl
from pathlib import Path
from argparse import Namespace
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import torch.distributed as dist
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

# Initialize FastAPI app
app = FastAPI()
engine: Optional[AsyncLLMEngine] = None

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

TIMEOUT_KEEP_ALIVE = 5  # seconds.

@app.get("/")
async def serve_home(request: Request):
    """Serves the index.html file."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/generate")
async def generate(request: Request) -> Response:
    """Handles text generation requests."""
    request_dict = await request.json()
    return await _generate(request_dict)

async def _generate(request_dict: dict) -> Response:
    """Handles token-by-token streaming response."""
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    # Optimize max output length for streaming performance
    max_tokens = request_dict.get("max_tokens", 512)  # Default to 512 for streaming
    sampling_params = SamplingParams(max_tokens=max_tokens, **request_dict)
    request_id = random_uuid()

    assert engine is not None, "Engine is not initialized!"
    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        """Streams tokens one-by-one."""
        async for request_output in results_generator:
            for output in request_output.outputs:
                chunk = json.dumps({"text": output.text}) + "\n"
                yield chunk.encode("utf-8")

    if stream:
        return StreamingResponse(stream_results(), media_type="application/json")

    # Non-streaming case: Return full output after processing
    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]
    return JSONResponse({"text": text_outputs})

async def init_engine(model_dir: str):
    """Initialize the vLLM engine with the specified model directory."""
    global engine
    engine_args = AsyncEngineArgs(
        model=model_dir,
        max_model_len=8192,  # Limit model max length
        gpu_memory_utilization=0.98  # Use 98% of GPU memory
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

def cleanup():
    """Cleanup function to destroy process groups on exit."""
    if dist.is_initialized():
        dist.destroy_process_group()

def run_server(model_dir: str, port: int = 8000):
    """Starts the FastAPI server."""
    asyncio.run(init_engine(model_dir))

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    finally:
        cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vLLM Streaming API")
    parser.add_argument("--model-dir", type=str, default="./deepseek-r1-awq", help="Path to the model directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    run_server(args.model_dir, args.port)
