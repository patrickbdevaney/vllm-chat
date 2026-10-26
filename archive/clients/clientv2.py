import json
import asyncio
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import AsyncGenerator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"
TIMEOUT = httpx.Timeout(30.0, connect=5.0)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>vLLM Chat Interface</title>
    <style>
        body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
        #response { min-height: 100px; overflow-y: auto; max-height: 500px; }
        textarea, #response { width: 100%; margin: 10px 0; }
        button { padding: 10px 20px; }
        #stats { font-size: 0.9em; color: gray; margin-top: 10px; }
        .error { color: red; }
        .loading { opacity: 0.6; }
    </style>
    <script>
        const controller = new AbortController();
        
        async function sendPrompt() {
            const prompt = document.getElementById("prompt").value;
            const responseElement = document.getElementById("response");
            const statsElement = document.getElementById("stats");
            const button = document.querySelector('button');
            
            responseElement.textContent = "";
            statsElement.textContent = "";
            button.disabled = true;
            responseElement.classList.add('loading');
            
            let totalTokens = 0;
            let startTime = Date.now();
            
            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ prompt: prompt }),
                    signal: controller.signal
                });
                
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                // Process tokens immediately without buffering
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    // Immediately process and display each chunk
                    const text = decoder.decode(value, { stream: true });
                    if (text) {
                        responseElement.textContent += text;
                        totalTokens += 1; // Increment by 1 since we're getting individual tokens
                        
                        // Update stats and scroll
                        const elapsedTime = (Date.now() - startTime) / 1000;
                        const tokensPerSecond = (totalTokens / elapsedTime).toFixed(2);
                        statsElement.textContent = `Tokens: ${totalTokens} | Time: ${elapsedTime.toFixed(2)}s | Tokens/sec: ${tokensPerSecond}`;
                        responseElement.scrollTop = responseElement.scrollHeight;
                    }
                }
                
            } catch (err) {
                if (err.name === 'AbortError') {
                    responseElement.innerHTML += "\n\n[Generation cancelled]";
                } else {
                    responseElement.innerHTML += `\n\n<span class="error">Error: ${err.message}</span>`;
                }
            } finally {
                button.disabled = false;
                responseElement.classList.remove('loading');
            }
        }
        
        function cancelGeneration() {
            controller.abort();
        }
    </script>
</head>
<body>
    <h1>vLLM Chat</h1>
    <textarea id="prompt" rows="4" cols="50" placeholder="Enter your prompt here"></textarea><br>
    <button onclick="sendPrompt()">Send</button>
    <button onclick="cancelGeneration()">Cancel</button>
    <pre id="response" style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;"></pre>
    <div id="stats"></div>
</body>
</html>
"""

async def create_async_client():
    return httpx.AsyncClient(timeout=TIMEOUT)

async def process_stream(line: str) -> str:
    """Process a single line from the stream and extract content."""
    try:
        if not line or not line.startswith('data: '):
            return ''
            
        line = line[6:]  # Remove 'data: ' prefix
        if line == "[DONE]":
            return ''
            
        data = json.loads(line)
        if 'choices' in data and data['choices']:
            content = data['choices'][0].get('delta', {}).get('content', '')
            return content
            
    except json.JSONDecodeError:
        logger.warning(f"Failed to decode JSON: {line}")
    except Exception as e:
        logger.error(f"Error processing stream: {e}")
        
    return ''

async def stream_response(response: httpx.Response) -> AsyncGenerator[bytes, None]:
    """Stream tokens immediately without buffering."""
    try:
        # Process each line (token) as soon as it arrives
        async for line in response.aiter_lines():
            content = await process_stream(line)
            if content:
                # Yield immediately without any sleep delay
                yield content.encode('utf-8')
                
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"\nError during streaming: {str(e)}".encode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        
        async with await create_async_client() as client:
            response = await client.post(
                VLLM_API_URL,
                json={
                    "model": "deepseek-r1-awq",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    # Add streaming optimization parameters if supported by your vLLM version
                    "temperature": 1.0,
                    "max_tokens": None  # Allow dynamic response length
                },
                headers={
                    "Accept": "text/event-stream",
                    "Connection": "keep-alive",
                    "Cache-Control": "no-cache"
                }
            )
            
            return StreamingResponse(
                stream_response(response),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Transfer-Encoding": "chunked"
                }
            )
            
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return StreamingResponse(
            iter([f"Error: {str(e)}".encode('utf-8')]),
            media_type="text/plain"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8500, 
        loop="asyncio",
        timeout_keep_alive=30,
        limit_concurrency=100
    )