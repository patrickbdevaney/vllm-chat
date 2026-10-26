# Archive

Earlier iterations, kept for history. Nothing here is maintained or needed to run the current apps in [`../app/`](../app/).

| Path | What it was | Superseded by |
|---|---|---|
| `servers/apiv2.py`, `servers/api-chat-server.py` (identical) | First in-process server: DeepSeek-R1-Distill-Qwen-7B with bitsandbytes 4-bit, 4K context. It served `templates/index.html`, which only exists in `streaming-attempts/` | `app/server.py` |
| `servers/inf.py` | Terminal REPL with the same 7B bitsandbytes setup | `app/server.py` |
| `servers/api32b.py`, `servers/api32bwebui.py` | API-only and web-UI servers for `inarikami/DeepSeek-R1-Distill-Qwen-32B-AWQ` (desktop GPU) | `app/server.py` with `VLLM_MODEL` |
| `servers/apicotstepwise.py` | Multi-round continuation returned through a `Response` with metric headers | `app/server.py` |
| `clients/client.py` … `clientv5.py` | Successive streaming front ends for `vllm serve` (sync `requests`, then `aiohttp`, then abort, stats and latency) | `app/stream_ui.py` (was `clientv6.py`) |
| `streaming-attempts/` | Hand-built token streaming from an in-process `LLM`. Abandoned because it duplicated text under concurrency | `vllm serve` + `app/stream_ui.py` |
| `static/styles.css` | Stylesheet used only by `streaming-attempts/index.html` | inline styles in `app/templates/index.html` |
