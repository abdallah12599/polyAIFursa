# Vision Agent

A LangChain-powered AI vision agent with a manual ReAct loop. Accepts text and base64-encoded images, and can call tools (e.g. YOLO object detection) to answer questions.

## Prerequisites

- Python 3.10+
- A running YOLO service (optional - only needed for `detect_objects`)


## Setup

Install dependencies (from `services/agent/`):

```bash
pip install -r requirements.txt
```

Configure environment:

```bash
cp .env.example .env
# Edit .env and set at least OPENAI_API_KEY (or another provider key) and MODEL
```

`.env` variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | - | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | - | Required for Anthropic models |
| `GOOGLE_API_KEY` | - | Required for Google models |
| `MODEL` | `claude-sonnet-4-6` | Any model string supported by `init_chat_model` |
| `YOLO_SERVICE_URL` | `http://localhost:8080` | URL of the YOLO microservice |

## Running

```bash
cd services/agent
python app.py
```

The server starts at `http://localhost:8000`.

## Testing with curl

### Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok"}
```

### Plain text message

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! What can you do?"}'
```

### Send a message with an image

```bash
echo "{\"message\": \"What objects are in this image?\", \"image_base64\": \"$(base64 -w0 beatles.jpeg)\"}" \
  | curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d @-
```

## API Reference

### `POST /chat`

Request body:

```json
{
  "message": "string (optional, defaults to 'What's in this image?')",
  "image_base64": "string (optional, base64-encoded JPEG or PNG)"
}
```

Response:

```json
{
  "response": "string",
  "prediction_id": "a1b2c3... or null",
  "annotated_image": "<base64 JPEG> or null",
  "agent_loop_time_s": 1.84,
  "iterations": 2,
  "tools_called": ["detect_objects"],
  "context_limit_exceeded": false,
  "tokens_used": { "input": 312, "output": 22, "total": 334 }
}
```

`annotated_image` is the YOLO bounding-box image (base64) and is populated
whenever a detection ran during the request.

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

## Capability checks & rate limiting

On startup the service validates the selected `MODEL` against its
[model profile](https://github.com/sst/models.dev): it requires `tool_calling`
and `structured_output`, and reads `max_input_tokens` to flag requests that
approach the context window (`context_limit_exceeded`). A LangChain
`InMemoryRateLimiter` (~0.5 req/s, burst 5) throttles calls so we stay under
provider rate limits.

## Testing

No real LLM or YOLO calls are made (the loop and tools are mocked):

```bash
cd services/agent
pip install -r requirements.txt
pytest tests/ -v
```
