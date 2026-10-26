from vllm import LLM, SamplingParams
import time

# Model name
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Load the model with optimized settings
llm = LLM(
    model=MODEL_NAME,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    gpu_memory_utilization=0.98,  # Use almost all available VRAM
    max_model_len=4096  # Reduce to fit within GPU memory
)

# Sampling parameters
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)

def chat_with_model():
    print("Welcome to the vLLM Chat Interface!")
    print("Type your message and press Enter. Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Measure start time
        start_time = time.time()

        # Generate response
        outputs = llm.generate([user_input], sampling_params)
        model_response = outputs[0].outputs[0].text

        # Performance metrics
        generation_time = time.time() - start_time
        num_tokens = len(outputs[0].outputs[0].token_ids)
        tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0

        # Display response
        print(f"\nModel: {model_response}\n")
        print(f"---\nMetrics: {generation_time:.2f}s | Tokens: {num_tokens} | Speed: {tokens_per_second:.2f} tok/sec\n---")

if __name__ == "__main__":
    chat_with_model()
