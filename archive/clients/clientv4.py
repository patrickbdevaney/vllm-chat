import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI()

# URL where the vLLM API server is running
VLLM_API_URL = "http://localhost:8000/v1/chat/completions"

# Modified HTML with better styling, error handling, and stats tracking
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>vLLM Chat Interface</title>
    <style>
        body { max-width: 800px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
        #response { min-height: 100px; }
        textarea, #response { width: 100%; margin: 10px 0; }
        button { padding: 10px 20px; }
        #stats { font-size: 0.9em; color: gray; margin-top: 10px; }
    </style>
    <script>
        async function sendPrompt() {
            const prompt = document.getElementById("prompt").value;
            const responseElement = document.getElementById("response");
            const statsElement = document.getElementById("stats");
            responseElement.textContent = "";
            statsElement.textContent = "";

            let totalTokens = 0;
            let startTime = Date.now();

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ prompt: prompt })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let previousContent = "";
                let tokensPerSecond = 0;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const content = decoder.decode(value);
                    responseElement.textContent += content;

                    // Estimate total tokens (simple count of characters for now)
                    const tokens = content.split(/\s+/).length;
                    totalTokens += tokens;

                    // Calculate tokens per second
                    const elapsedTime = (Date.now() - startTime) / 1000; // in seconds
                    tokensPerSecond = (totalTokens / elapsedTime).toFixed(2);

                    // Update stats
                    statsElement.textContent = `Tokens: ${totalTokens} | Time: ${elapsedTime.toFixed(2)}s | Tokens/sec: ${tokensPerSecond}`;

                }
            } catch (err) {
                responseElement.textContent = "Error: " + err;
            }
        }
    </script>
</head>
<body>
    <h1>vLLM Chat</h1>
    <textarea id="prompt" rows="4" cols="50" placeholder="Enter your prompt here"></textarea><br>
    <button onclick="sendPrompt()">Send</button>
    <pre id="response" style="border: 1px solid #ccc; padding: 10px; white-space: pre-wrap;"></pre>
    <div id="stats"></div> <!-- Stats display -->
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

def post_http_request(prompt: str, stream: bool = True) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": "deepseek-r1-awq",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }
    response = requests.post(VLLM_API_URL, headers=headers, json=pload, stream=True)
    return response

def stream_generator(response: requests.Response):
    """
    Process the streaming response and yield only the content.
    Handles the 'data: ' prefix in the response and properly extracts content.
    """
    for line in response.iter_lines(chunk_size=4, decode_unicode=True):
        if line and line.startswith('data: '):
            try:
                # Remove the 'data: ' prefix and handle [DONE]
                line = line[6:]  # Skip "data: "
                if line == "[DONE]":
                    continue
                    
                # Parse the JSON data
                data = json.loads(line)
                
                # Extract content from the delta
                if 'choices' in data and len(data['choices']) > 0:
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    
                    # Only yield if there's actual content
                    if content:
                        yield content.encode('utf-8')
                        
            except json.JSONDecodeError:
                continue  # Skip any malformed JSON
            except Exception as e:
                # Log the error but don't send it to the client
                print(f"Error processing line: {e}")
                continue

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    response = post_http_request(prompt, stream=True)
    return StreamingResponse(stream_generator(response), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
