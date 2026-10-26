import json
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import aiohttp

app = FastAPI()

VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>vLLM Chat Interface</title>
    <style>
        body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
        #response { min-height: 100px; white-space: pre-wrap; }
        textarea, #response { width: 100%; margin: 10px 0; }
        button { padding: 10px 20px; }
        #stats { font-size: 0.9em; color: gray; margin-top: 10px; }
        .loading { opacity: 0.6; }
    </style>
    <script>
        let controller = null;

        async function sendPrompt() {
            const prompt = document.getElementById("prompt").value;
            const responseElement = document.getElementById("response");
            const statsElement = document.getElementById("stats");
            const button = document.querySelector('button');
            
            // Reset UI
            responseElement.textContent = "";
            statsElement.textContent = "";
            
            // Abort previous request if exists
            if (controller) {
                controller.abort();
            }
            
            // Create new abort controller
            controller = new AbortController();
            
            // Update UI state
            button.classList.add('loading');
            button.disabled = true;
            
            let totalTokens = 0;
            let startTime = Date.now();
            
            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ prompt: prompt }),
                    signal: controller.signal
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {
                    const {value, done} = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value, {stream: true});
                    responseElement.textContent += chunk;
                    
                    // Auto-scroll to bottom
                    responseElement.scrollTop = responseElement.scrollHeight;
                    
                    // Update stats
                    const tokens = chunk.split(/\s+/).length;
                    totalTokens += tokens;
                    const elapsedTime = (Date.now() - startTime) / 1000;
                    const tokensPerSecond = (totalTokens / elapsedTime).toFixed(2);
                    statsElement.textContent = 
                        `Tokens: ${totalTokens} | Time: ${elapsedTime.toFixed(2)}s | Tokens/sec: ${tokensPerSecond}`;
                }
            } catch (err) {
                if (err.name === 'AbortError') {
                    responseElement.textContent += "\\n[Request cancelled]";
                } else {
                    responseElement.textContent += "\\nError: " + err.message;
                }
            } finally {
                button.classList.remove('loading');
                button.disabled = false;
                controller = null;
            }
        }
    </script>
</head>
<body>
    <h1>vLLM Chat</h1>
    <textarea id="prompt" rows="4" cols="50" placeholder="Enter your prompt here"></textarea><br>
    <button onclick="sendPrompt()">Send</button>
    <pre id="response" style="border: 1px solid #ccc; padding: 10px;"></pre>
    <div id="stats"></div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE

async def stream_generator(prompt: str):
    """
    Enhanced streaming generator that yields each token as soon as it's received.
    """
    async with aiohttp.ClientSession() as session:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "deepseek-r1-awq",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        try:
            async with session.post(VLLM_API_URL, json=payload, headers=headers) as response:
                # Immediately stream data as it arrives
                async for line in response.content:
                    if line:
                        line = line.decode().strip()
                        if line.startswith('data: '):
                            line = line[6:]  # Remove 'data: ' prefix
                            if line == "[DONE]":
                                continue
                            
                            try:
                                data = json.loads(line)
                                if 'choices' in data and data['choices']:
                                    content = data['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        # Yield immediately without buffering
                                        yield content.encode('utf-8')
                            except json.JSONDecodeError:
                                continue
        except aiohttp.ClientError as e:
            yield f"\nConnection error: {str(e)}".encode('utf-8')
        except Exception as e:
            yield f"\nUnexpected error: {str(e)}".encode('utf-8')

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        
        return StreamingResponse(
            stream_generator(prompt),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked"
            }
        )
    except Exception as e:
        return StreamingResponse(
            iter([f"Error: {str(e)}".encode('utf-8')]),
            media_type="text/plain"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
