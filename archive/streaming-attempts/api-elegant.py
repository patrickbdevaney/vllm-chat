import asyncio
import json
import logging
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

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


logger = logging.getLogger(__name__)

async def _generate(request_dict: dict) -> Response:
    """Handles token-by-token streaming response with improved batching and deduplication."""
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    max_tokens = min(request_dict.get("max_tokens", 2048), 2048)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=request_dict.get("temperature", 0.7),
        top_p=request_dict.get("top_p", 0.9),
        frequency_penalty=request_dict.get("frequency_penalty", 0.3),
        n=1,
        **request_dict
    )
    request_id = random_uuid()

    assert engine is not None, "Engine is not initialized!"
    results_generator = engine.generate(prompt, sampling_params, request_id)

    seen_ngrams = set()  # ✅ Store seen N-grams to avoid repetition
    buffer = []  # ✅ Holds collected words until a sentence is formed
    token_count = 0  # ✅ Tracks token count for logging
    N_GRAM_SIZE = 5  # ✅ Defines the N-gram size for duplication detection

    logger.info(f"📩 Received request: {request_dict}")
    logger.info(f"📝 Prompt: {prompt}")

    async def stream_results() -> AsyncGenerator[bytes, None]:
        """Streams text chunks while ensuring no duplicate N-grams."""
        nonlocal buffer, token_count

        async for request_output in results_generator:
            for output in request_output.outputs:
                new_text = output.text.strip()

                if not new_text:
                    continue  # ✅ Ignore empty outputs

                buffer.append(new_text)

                # ✅ Check for sentence completion
                if any(new_text.endswith(end) for end in [".", "!", "?", "\n"]):
                    full_chunk = " ".join(buffer).strip()
                    buffer = []  # ✅ Reset buffer after sending

                    # ✅ Generate N-grams from the full chunk
                    words = full_chunk.split()
                    n_grams = {" ".join(words[i:i + N_GRAM_SIZE]) for i in range(len(words) - N_GRAM_SIZE + 1)}

                    # ✅ Avoid duplicate N-grams
                    if not seen_ngrams.intersection(n_grams):
                        seen_ngrams.update(n_grams)  # ✅ Track new N-grams

                        if token_count < 10_000:
                            logger.info(f"🔹 Streaming: {full_chunk[:500]}...")
                        token_count += len(words)

                        yield json.dumps({"text": full_chunk}).encode("utf-8") + b"\n"

            await asyncio.sleep(0.05)  # ✅ Adjust streaming speed

        if buffer:  # ✅ Send any remaining buffered content
            full_chunk = " ".join(buffer).strip()
            words = full_chunk.split()
            n_grams = {" ".join(words[i:i + N_GRAM_SIZE]) for i in range(len(words) - N_GRAM_SIZE + 1)}

            if not seen_ngrams.intersection(n_grams):
                seen_ngrams.update(n_grams)

                if token_count < 10_000:
                    logger.info(f"🔹 Streaming (Final): {full_chunk[:500]}...")
                yield json.dumps({"text": full_chunk}).encode("utf-8") + b"\n"

    if stream:
        return StreamingResponse(stream_results(), media_type="application/json")

    final_output = None
    async for request_output in results_generator:
        final_output = request_output

    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs]

    logger.info(f"✅ Full Completion (capped at 10,000 tokens logged)")
    return JSONResponse({"text": text_outputs})




async def init_engine(model_dir: str):
    """Initialize the vLLM engine with the specified model directory."""
    global engine
    engine_args = AsyncEngineArgs(
        model=model_dir,
        max_model_len=2048,  # ✅ **Restrict max model length**
        gpu_memory_utilization=0.98  # ✅ **Use 98% of GPU memory**
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    logger.info("🚀 LLM Engine Initialized")

def cleanup():
    """Cleanup function to destroy process groups on exit."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("🛑 Distributed Process Group Destroyed")

async def main(model_dir: str, port: int):
    """Async entrypoint to start the FastAPI server."""
    await init_engine(model_dir)
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    logger.info(f"🌐 Server running at http://localhost:{port}")
    await server.serve()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="vLLM Streaming API")
    parser.add_argument("--model-dir", type=str, default="./deepseek-r1-awq", help="Path to the model directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    asyncio.run(main(args.model_dir, args.port))
