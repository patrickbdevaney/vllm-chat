import json
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

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
            let buffer = ''; // Buffer for partial UTF-8 characters

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({ prompt: prompt })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8', { fatal: false });

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    // Decode the incoming bytes and add to buffer
                    buffer += decoder.decode(value, { stream: true });
                    
                    // Process complete words/sentences
                    const completeText = buffer.split(/(?<=[.!?\n])\s+/);
                    
                    if (completeText.length > 1) {
                        // Join all complete sentences except the last (potentially incomplete) one
                        const toDisplay = completeText.slice(0, -1).join(' ');
                        responseElement.textContent += toDisplay + ' ';
                        
                        // Keep the last (potentially incomplete) sentence in buffer
                        buffer = completeText[completeText.length - 1];
                        
                        // Update stats immediately
                        const tokens = toDisplay.split(/\s+/).length;
                        totalTokens += tokens;
                        const elapsedTime = (Date.now() - startTime) / 1000;
                        const tokensPerSecond = (totalTokens / elapsedTime).toFixed(2);
                        statsElement.textContent = `Tokens: ${totalTokens} | Time: ${elapsedTime.toFixed(2)}s | Tokens/sec: ${tokensPerSecond}`;
                    }
                }
                
                // Display any remaining text in buffer
                if (buffer) {
                    responseElement.textContent += buffer;
                    const tokens = buffer.split(/\s+/).length;
                    totalTokens += tokens;
                    const elapsedTime = (Date.now() - startTime) / 1000;
                    const tokensPerSecond = (totalTokens / elapsedTime).toFixed(2);
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
    <div id="stats"></div>
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

async def stream_generator(response: requests.Response):
    """
    Process the streaming response and yield content immediately.
    """
    for line in response.iter_lines(chunk_size=1, decode_unicode=False):
        if line:
            try:
                # Convert bytes to string and handle the 'data: ' prefix
                line_str = line.decode('utf-8')
                if not line_str.startswith('data: '):
                    continue
                    
                line_str = line_str[6:]  # Remove 'data: ' prefix
                if line_str == "[DONE]":
                    continue
                    
                # Parse the JSON data
                data = json.loads(line_str)
                
                # Extract and yield content immediately
                if 'choices' in data and data['choices']:
                    delta = data['choices'][0].get('delta', {})
                    content = delta.get('content', '')
                    if content:
                        yield content.encode('utf-8')
                        
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
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