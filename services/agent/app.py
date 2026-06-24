import base64
import io
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Optional
import time

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logging.getLogger("langchain").setLevel(logging.DEBUG)
logging.getLogger("langchain_core").setLevel(logging.DEBUG)
log = logging.getLogger("agent")

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080")
MODEL = os.environ.get("MODEL")

# Text-only models
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5","google_genai:gemini-2.5-flash"
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported text-only models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

# Flag a request once its prompt uses this fraction of the model's context window.
CONTEXT_LIMIT_RATIO = 0.9

# Image data is passed to the YOLO tool out-of-band, never inside the messages
# the LLM sees (the model's job is conversation, not vision).
_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)

# The YOLO tool records the prediction id + annotated image here so the agent
# can return them WITHOUT ever putting the image into an LLM message.
_last_detection: ContextVar[Optional[dict]] = ContextVar("last_detection", default=None)


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class AgentResult(BaseModel):
    """Everything the agentic loop produced for one /chat request."""
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None       # base64 JPEG, never shown to the LLM
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    image_bytes = base64.b64decode(image_b64)
    with httpx.Client(timeout=30.0) as client:
        predict_resp = client.post(
            f"{YOLO_SERVICE_URL}/predict",
            files={"file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")},
        )
        predict_resp.raise_for_status()
        data = predict_resp.json()

        # Fetch the annotated (bounding-box) image so the agent can return it.
        uid = data.get("prediction_uid")
        annotated_b64 = None
        if uid:
            try:
                img_resp = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}/image")
                img_resp.raise_for_status()
                annotated_b64 = base64.b64encode(img_resp.content).decode("ascii")
            except httpx.HTTPError as exc:
                log.warning("Could not fetch annotated image for %s: %s", uid, exc)

    # Keep image + id out of the LLM-visible text.
    _last_detection.set({"prediction_id": uid, "annotated_image": annotated_b64})

    # Return text-only detection facts to the model.
    return json.dumps({
        "prediction_uid": uid,
        "detection_count": data.get("detection_count"),
        "labels": data.get("labels"),
        "time_took": data.get("time_took"),
    })


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects
}

# Throttle outbound LLM calls to stay under provider rate limits (~0.5 req/s, burst 5).
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=5,
)

llm = init_chat_model(MODEL, temperature=0, rate_limiter=rate_limiter)


def _model_profile() -> dict:
    """Best-effort read of the model capability profile (powered by models.dev)."""
    try:
        return llm.profile or {}
    except Exception as exc:  # profile data may be unavailable for some models
        log.warning("Could not read model profile for %s: %s", MODEL, exc)
        return {}


_profile = _model_profile()


def _require_capability(name: str) -> None:
    # Only fail when the profile *explicitly* says the capability is missing.
    if _profile.get(name) is False:
        raise SystemExit(
            f"\n[ERROR] MODEL='{MODEL}' does not support required capability '{name}'.\n"
            f"Choose a model whose profile reports '{name}': true.\n"
        )


_require_capability("tool_calling")
_require_capability("structured_output")

# Max prompt size for the chosen model (None if unknown). Used to flag requests
# that get close to the context window.
MAX_INPUT_TOKENS = _profile.get("max_input_tokens")

llm_with_tools = llm.bind_tools(list(TOOLS.values()))


def _content_to_text(content) -> str:
    """
    Normalize an AIMessage's content to a plain string.

    Some providers (e.g. Anthropic) return content as a list of typed blocks
    like [{"type": "text", "text": "..."}] instead of a bare string, so we
    join the text blocks together.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def run_agent(history: list, max_iterations: int = 10) -> AgentResult:
    """
    Manual ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response or we hit
         `max_iterations` (the guard against an endlessly tool-calling model).
    """
    start = time.perf_counter()
    _last_detection.set(None)
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history
    iterations = 0

    iterations = 0
    tools_called: list[str] = []
    tokens = TokenUsage()
    context_limit_exceeded = False

    while iterations < max_iterations:
        iterations += 1
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            tokens = TokenUsage(
                input=usage.get("input_tokens", 0),
                output=usage.get("output_tokens", 0),
                total=usage.get("total_tokens", 0),
            )
            if MAX_INPUT_TOKENS and usage.get("input_tokens", 0) >= CONTEXT_LIMIT_RATIO * MAX_INPUT_TOKENS:
                context_limit_exceeded = True

        # No tool calls -> the model produced its final answer.
        if not response.tool_calls:
            return _build_result(
                response=_content_to_text(response.content),
                time_s=start,
                iterations=iterations,
                tools_called=tools_called,
                context_limit_exceeded=context_limit_exceeded,
                tokens=tokens,
            )

        # Execute every tool the model requested.
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            tools_called.append(name)
            tool_fn = TOOLS.get(name)
            if tool_fn is None:
                messages.append(ToolMessage(
                    content=json.dumps({"error": f"Unknown tool '{name}'."}),
                    tool_call_id=tool_call["id"],
                ))
                continue
            try:
                messages.append(tool_fn.invoke(tool_call))
            except Exception as exc:
                log.warning("Tool '%s' failed: %s", name, exc)
                messages.append(ToolMessage(
                    content=json.dumps({"error": f"Tool '{name}' failed: {exc}"}),
                    tool_call_id=tool_call["id"],
                ))

    # Loop exhausted without a final answer.
    log.warning("run_agent hit max_iterations=%s", max_iterations)
    return _build_result(
        response="I couldn't complete your request within the allowed number of steps. "
                 "Please try again or rephrase your question.",
        time_s=start,
        iterations=iterations,
        tools_called=tools_called,
        context_limit_exceeded=context_limit_exceeded,
        tokens=tokens,
    )


def _build_result(response, time_s, iterations, tools_called, context_limit_exceeded, tokens) -> AgentResult:
    detection = _last_detection.get() or {}
    return AgentResult(
        response=response,
        prediction_id=detection.get("prediction_id"),
        annotated_image=detection.get("annotated_image"),
        agent_loop_time_s=round(time.perf_counter() - time_s, 2),
        iterations=iterations,
        tools_called=tools_called,
        context_limit_exceeded=context_limit_exceeded,
        tokens_used=tokens,
    )

app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str                           # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]         # full conversation thread, oldest first


@app.post("/chat", response_model=AgentResult)
def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64          #) saved for detect_objects tool
                content = msg.content + "\n[An image was uploaded. Use existing tools to analyze it according to user instructions.]"
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    start_time = time.time()

    token = _current_image_b64.set(latest_image)
    try:
        return run_agent(lc_messages)
    except Exception as exc:
        log.exception("Agent run failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}")
    finally:
        _current_image_b64.reset(token)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
