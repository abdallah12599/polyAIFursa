import base64
import io
import json
import logging
import os
import time
from contextvars import ContextVar
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent")

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import tool
from pydantic import BaseModel, Field

YOLO_SERVICE_URL = os.environ.get("YOLO_SERVICE_URL", "http://localhost:8080").rstrip("/")
MODEL = (os.environ.get("MODEL") or "").strip()

# Models we have validated for this service. Capabilities are still verified at
# runtime against the model profile (see `_require_capabilities`).
ALLOWED_MODELS = {
    "openai:gpt-5.4-mini",
    "anthropic:claude-haiku-4-5",
    "google_genai:gemini-2.5-flash",
}

if MODEL not in ALLOWED_MODELS:
    allowed_list = "\n  ".join(sorted(ALLOWED_MODELS))
    raise SystemExit(
        f"\n[ERROR] MODEL='{MODEL}' is not allowed.\n"
        f"Set MODEL in your .env to one of the supported models:\n  {allowed_list}\n"
    )

SYSTEM_PROMPT = (
    "You are an AI vision assistant. You help users understand and analyze images. "
    "Use the available tools to extract information from images. "
)

# How close to the model's context window we allow before flagging the request.
CONTEXT_LIMIT_RATIO = 0.9

# Carries the current request's image and any prediction ids produced by tools,
# so tools don't need them passed as LLM-visible arguments.
_current_image_b64: ContextVar[Optional[str]] = ContextVar("current_image_b64", default=None)
_prediction_uids: ContextVar[list] = ContextVar("prediction_uids", default=[])


@tool
def detect_objects() -> str:
    """Detect and identify objects in the image provided by the user using YOLO object detection."""
    image_b64 = _current_image_b64.get()
    if not image_b64:
        return json.dumps({"error": "No image was provided by the user."})

    try:
        image_bytes = base64.b64decode(image_b64)
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{YOLO_SERVICE_URL}/predict",
                files={"file": ("image.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            )
            response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # network / decoding / YOLO errors
        logger.exception("detect_objects failed")
        return json.dumps({"error": f"Object detection failed: {exc}"})

    uid = payload.get("prediction_uid")
    if uid:
        _prediction_uids.set(_prediction_uids.get() + [uid])
    return json.dumps(payload)


# Registry: map tool name -> tool function
TOOLS = {
    detect_objects.name: detect_objects,
}


def _build_llm():
    """Initialise the chat model with a shared in-memory rate limiter."""
    # Realistic limit that fits the free/low tiers of OpenAI, Anthropic and
    # Google: ~30 requests/min, with a small burst allowance.
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5,
        check_every_n_seconds=0.1,
        max_bucket_size=5,
    )
    return init_chat_model(MODEL, temperature=0, rate_limiter=rate_limiter)


def _require_capabilities(model) -> dict:
    """
    Validate the selected model supports the features this agent needs.
    Returns the profile dict (possibly empty if the provider exposes none).
    """
    profile = getattr(model, "profile", None) or {}
    if not profile:
        logger.warning("No model profile available for %s; skipping capability checks.", MODEL)
        return {}

    missing = []
    if not profile.get("tool_calling"):
        missing.append("tool_calling")
    # Some providers name this 'structured_output'; tolerate either.
    if not (profile.get("structured_output") or profile.get("structured_outputs")):
        missing.append("structured_output")
    if missing:
        raise SystemExit(
            f"\n[ERROR] Model '{MODEL}' is missing required capabilities: "
            f"{', '.join(missing)}.\nChoose a model that supports tool calling and structured output.\n"
        )

    logger.info(
        "Model %s OK | max_input_tokens=%s tool_calling=%s structured_output=%s",
        MODEL,
        profile.get("max_input_tokens"),
        profile.get("tool_calling"),
        profile.get("structured_output") or profile.get("structured_outputs"),
    )
    return profile


llm = _build_llm()
MODEL_PROFILE = _require_capabilities(llm)
MAX_INPUT_TOKENS = MODEL_PROFILE.get("max_input_tokens")
llm_with_tools = llm.bind_tools(list(TOOLS.values()))


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class AgentResult(BaseModel):
    """Everything the agentic loop produces, beyond the plain text answer."""
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None  # base64-encoded JPEG, or None
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)


def _fetch_annotated_image(uid: str) -> Optional[str]:
    """Fetch the annotated (bounding-box) image from YOLO and base64-encode it."""
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{YOLO_SERVICE_URL}/prediction/{uid}/image")
            resp.raise_for_status()
        return base64.b64encode(resp.content).decode("ascii")
    except Exception:
        logger.exception("Could not fetch annotated image for %s", uid)
        return None


def run_agent(history: list, max_iterations: int = 10) -> AgentResult:
    """
    ReAct loop:
      1. Send messages to the LLM.
      2. If the LLM requests tool calls, execute them and append results.
      3. Repeat until the LLM returns a plain text response, or `max_iterations`
         is reached (guard against a model that calls tools forever).
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + history

    result = AgentResult(response="")
    started = time.monotonic()

    for iteration in range(1, max_iterations + 1):
        result.iterations = iteration
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        # Accumulate token usage across the whole loop.
        usage = getattr(response, "usage_metadata", None) or {}
        result.tokens_used.input += int(usage.get("input_tokens", 0) or 0)
        result.tokens_used.output += int(usage.get("output_tokens", 0) or 0)
        result.tokens_used.total += int(usage.get("total_tokens", 0) or 0)

        # Detect when we're approaching the model's context window.
        if MAX_INPUT_TOKENS and usage.get("input_tokens"):
            if usage["input_tokens"] >= CONTEXT_LIMIT_RATIO * MAX_INPUT_TOKENS:
                result.context_limit_exceeded = True
                logger.warning(
                    "Input tokens %s near max %s", usage["input_tokens"], MAX_INPUT_TOKENS
                )

        # No tool calls -> the model produced its final answer.
        if not response.tool_calls:
            result.response = response.content if isinstance(response.content, str) else str(response.content)
            break

        # Execute every tool the model requested.
        for tool_call in response.tool_calls:
            name = tool_call["name"]
            result.tools_called.append(name)
            tool_fn = TOOLS.get(name)
            if tool_fn is None:
                messages.append(
                    ToolMessage(
                        content=json.dumps({"error": f"Unknown tool '{name}'"}),
                        tool_call_id=tool_call["id"],
                    )
                )
                continue
            try:
                tool_result = tool_fn.invoke(tool_call)  # returns a ToolMessage
            except Exception as exc:
                logger.exception("Tool '%s' raised", name)
                tool_result = ToolMessage(
                    content=json.dumps({"error": f"Tool '{name}' failed: {exc}"}),
                    tool_call_id=tool_call["id"],
                )
            messages.append(tool_result)
    else:
        # Loop exhausted without a final text answer.
        result.response = (
            "I couldn't complete the request within the allowed number of steps. "
            "Please try rephrasing or simplifying your question."
        )
        logger.warning("run_agent hit max_iterations=%s", max_iterations)

    # Attach the annotated image from the most recent detection, if any.
    uids = _prediction_uids.get()
    if uids:
        result.prediction_id = uids[-1]
        result.annotated_image = _fetch_annotated_image(uids[-1])

    result.agent_loop_time_s = round(time.monotonic() - started, 3)
    return result


app = FastAPI(title="Vision Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    image_base64: Optional[str] = None  # only on user messages that carry an image


class ChatRequest(BaseModel):
    messages: list[ChatMessage]  # full conversation thread, oldest first


class ChatResponse(BaseModel):
    response: str
    prediction_id: Optional[str] = None
    annotated_image: Optional[str] = None
    agent_loop_time_s: float = 0.0
    iterations: int = 0
    tools_called: list[str] = Field(default_factory=list)
    context_limit_exceeded: bool = False
    tokens_used: TokenUsage = Field(default_factory=TokenUsage)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    lc_messages = []
    latest_image = None

    for msg in request.messages:
        if msg.role == "user":
            if msg.image_base64:
                latest_image = msg.image_base64  # saved for detect_objects tool
                content = msg.content + (
                    "\n[An image was uploaded. Use existing tools to analyze it "
                    "according to user instructions.]"
                )
            else:
                content = msg.content
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=msg.content))

    image_token = _current_image_b64.set(latest_image)
    uids_token = _prediction_uids.set([])
    try:
        result = run_agent(lc_messages)
    except Exception as exc:
        logger.exception("run_agent failed")
        raise HTTPException(status_code=502, detail=f"Agent error: {exc}")
    finally:
        _current_image_b64.reset(image_token)
        _prediction_uids.reset(uids_token)

    return ChatResponse(**result.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
