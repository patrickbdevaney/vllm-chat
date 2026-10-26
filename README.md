# vllm-chat

**A private, local chat front end for DeepSeek-R1-Distill on a consumer RTX laptop GPU, served with vLLM**

| | |
|---|---|
| **Purpose** | Personal chat over a local network. Single user, chat completions only; not a general inference or agent API |
| **Model** | AWQ-quantized [DeepSeek-R1-Distill-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B). The 7B fits the 8 GB of an RTX 4060 laptop GPU; the 32B AWQ build was used on a larger desktop GPU |
| **Engine** | vLLM, with the `deepseek_r1` reasoning parser |
| **Front ends** | FastAPI: an in-process server with full responses, or a streaming UI over vLLM's OpenAI-compatible server |
| **Hardware** | NVIDIA RTX 4060 Laptop GPU (8 GB) |
| **Status** | Archived personal project (2025-02). Consolidated and documented 2026-10 |

---

## Two ways to run it

| | In-process server | Streaming UI |
|---|---|---|
| Entry point | [`app/server.py`](app/server.py) | [`scripts/serve.sh`](scripts/serve.sh) + [`app/stream_ui.py`](app/stream_ui.py) |
| Processes | One: vLLM loaded inside FastAPI | Two: `vllm serve` (:8000) and the UI proxy (:8500) |
| Response | Whole answer at once, with token count, tok/s and wall time | Tokens stream as generated, with live tok/s and time to first chunk |
| Prompting | Fixed system prompt plus the user turn as raw text | Chat template applied by `vllm serve` (`/v1/chat/completions`) |
| UI | Chat history, copy buttons ([`app/templates/index.html`](app/templates/index.html)) | Single prompt and response pane, cancel on resend |
| Port | 8000 | 8500 (UI), 8000 (vLLM) |

**Why streaming goes through `vllm serve`.** A hand-built streaming loop around an in-process `LLM` duplicated text
under concurrent requests, sometimes repeating output thousands of times in the UI (`archive/streaming-attempts/`).
vLLM's own OpenAI-compatible server streams correctly, so the streaming UI is a thin proxy in front of it.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**In-process server**

```bash
python app/server.py                                  # http://localhost:8000
```

**Streaming UI**

```bash
./scripts/serve.sh                                    # terminal 1: vLLM OpenAI server on :8000
python app/stream_ui.py                               # terminal 2: http://localhost:8500
```

Both bind to `127.0.0.1` by default. To reach them from other devices on your network, set `HOST=0.0.0.0` (see
[Security](#security)).

---

## Configuration

All settings are environment variables.

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `VLLM_MODEL` | `server.py`, `serve.sh` | `casperhansen/deepseek-r1-distill-qwen-7b-awq` | Any HF id or local path. The original setup loaded a local AWQ directory, `./deepseek-r1-awq` |
| `MAX_MODEL_LEN` | `server.py`, `serve.sh` | `8192` | `4096` prefills faster and frees KV cache |
| `GPU_MEMORY_UTILIZATION` | `server.py`, `serve.sh` | `0.98` / `0.95` | Fraction of VRAM vLLM may claim; leave headroom if the GPU also drives a display |
| `SERVED_MODEL_NAME` | `serve.sh`, `stream_ui.py` | `deepseek-r1-awq` | Must match between the two |
| `VLLM_API_URL` | `stream_ui.py` | `http://localhost:8000/v1/chat/completions` | Upstream for the streaming UI |
| `ENABLE_REASONING_FLAG` | `serve.sh` | `--enable-reasoning` | Required by vLLM 0.7.x; set it empty on newer vLLM, which removed the flag |
| `HOST`, `PORT` | all | `127.0.0.1`; 8000 / 8500 | Bind address |

**Sampling (in-process server):** temperature 0.7, top-p 0.9, `max_tokens` = `MAX_MODEL_LEN`. If a response ever
fills that budget, the server feeds it back and continues, for at most 3 rounds.

---

## HTTP interface

| Server | Route | Method | Body | Returns |
|---|---|---|---|---|
| In-process | `/` | GET | — | Chat UI |
| In-process | `/chat` | POST | `{"user_input": "..."}` | `{"response", "tokens", "tokens_per_second", "generation_time"}` |
| Streaming UI | `/` | GET | — | Chat UI |
| Streaming UI | `/chat` | POST | `{"prompt": "..."}` | `text/plain` stream of content deltas |

`vllm serve` also exposes the standard OpenAI routes on :8000. With `--reasoning-parser deepseek_r1`, the model's
`<think>` content goes to `reasoning_content`. The streaming UI shows `content` deltas only.

---

## Performance

| Setup | Decode | Source |
|---|--:|---|
| vLLM, DeepSeek-R1-Distill-Qwen-7B, RTX 4060 Laptop | **~39 tok/s** | Original README (2025-02) |
| llama.cpp / ExLlamaV2 on the same laptop | ~32–35 tok/s | Original README (2025-02) |

These figures date from the original work. The exact quantization, prompt and engine versions were not recorded, so
treat them as indicative rather than a controlled benchmark. Both UIs report live tok/s, so you can measure your own
setup directly.

A 6-bit ExLlamaV2 quantization of the 7B distill, made for text-generation-webui, is on Hugging Face:
[`patrickbdevaney/deepseek-r1-qwen-7b-q6-exl2`](https://huggingface.co/patrickbdevaney/deepseek-r1-qwen-7b-q6-exl2).

---

## Validation

Both apps were smoke-tested on 2026-10-26 with current FastAPI (Starlette 1.10):
- **Streaming UI:** run against a stub OpenAI SSE server. Deltas stream through to `/chat`, and the page serves
  correctly.
- **In-process server:** run with vLLM stubbed out. `/` renders the template, `/chat` returns the metrics JSON, and
  an empty input returns 400.

The consolidation changed only configuration (environment variables, file-relative template path) and one Starlette
API call. The generation logic is the original's. A real-GPU run of the reorganized tree has not been repeated.

---

## Limitations

- **Single user.** No request queueing, sessions or conversation memory. Each request is one turn.
- **The in-process server does not apply the chat template.** It sends the system prompt and user text as raw text.
  The `vllm serve` path applies the model's template.
- **The streaming UI hides reasoning.** It shows `content` only, not `reasoning_content`.
- **No authentication** on any endpoint.
- **Unpinned dependencies.** The original ran on vLLM 0.7.x (early 2025). Newer vLLM removed `--enable-reasoning`
  (see `ENABLE_REASONING_FLAG`).

## Security

Nothing here authenticates requests. The defaults bind to localhost. If you set `HOST=0.0.0.0` to use it from a phone
or another machine, keep it on a trusted network. `vllm serve` also accepts `--api-key`.

## Repository layout

| Path | Contents |
|---|---|
| [`app/server.py`](app/server.py) | In-process FastAPI + vLLM server (full responses) |
| [`app/templates/index.html`](app/templates/index.html) | Chat UI for the in-process server |
| [`app/stream_ui.py`](app/stream_ui.py) | Streaming chat UI and proxy for `vllm serve` |
| [`scripts/serve.sh`](scripts/serve.sh) | vLLM OpenAI-compatible server launch with the reasoning parser |
| [`archive/`](archive/) | Earlier servers, clients and streaming attempts ([index](archive/README.md)) |

## License

- **Code:** MIT ([`LICENSE`](LICENSE)).
- **DeepSeek-R1-Distill models:** MIT (DeepSeek), built on Qwen2.5. Quantized checkpoints carry their publishers'
  terms.
- **vLLM:** Apache-2.0.
