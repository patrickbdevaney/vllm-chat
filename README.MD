# FastAPI LLM Chatbot

## Setup Instructions

1. **Create a virtual environment and install dependencies:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Navigate to the FastAPI project directory:**
   ```sh
   cd vllm-chat
   ```

3. **Run the FastAPI server:**
   ```sh
   python apiv2.py
   ```

4. **Access the chatbot:**
   - Open a browser and go to: `http://127.0.0.1:8000`
   - Start chatting with the LLM!

## Performance

This implementation achieves an output rate of **39 tokens/second** on an **RTX 4060**, outperforming Llama.cpp and ExLlamaV2, which typically run between **32-35 tokens/second**.

## Miscellaneous
You can also find my exl2 quantization on Huggingface:
https://huggingface.co/patrickbdevaney/deepseek-r1-qwen-7b-q6-exl2

It is easily used with text gen webui. You can also just download a GGUF of this model and run it on llama.cpp or text gen webui.
