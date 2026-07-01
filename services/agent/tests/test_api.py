"""API-layer tests for the agent service.

The entire agentic loop (`run_agent`) is mocked, so these tests exercise only
the FastAPI request/response handling - no LLM or YOLO calls are made.
"""
from fastapi.testclient import TestClient

import app as agent_app
from app import AgentResult, TokenUsage


client = TestClient(agent_app.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_full_response(monkeypatch):
    fake = AgentResult(
        response="There are 2 people in the image.",
        prediction_id="abc123",
        annotated_image="QkFTRTY0",
        agent_loop_time_s=1.23,
        iterations=2,
        tools_called=["detect_objects"],
        context_limit_exceeded=False,
        tokens_used=TokenUsage(input=312, output=22, total=334),
    )
    monkeypatch.setattr(agent_app, "run_agent", lambda history: fake)

    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "How many people?", "image_base64": "aW1n"}
            ]
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "There are 2 people in the image."
    assert data["prediction_id"] == "abc123"
    assert data["annotated_image"] == "QkFTRTY0"
    assert data["iterations"] == 2
    assert data["tools_called"] == ["detect_objects"]
    assert data["context_limit_exceeded"] is False
    assert data["tokens_used"] == {"input": 312, "output": 22, "total": 334}


def test_chat_handles_agent_failure(monkeypatch):
    def boom(history):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr(agent_app, "run_agent", boom)

    response = client.post(
        "/chat",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    assert "Agent error" in response.json()["detail"]


def test_chat_rejects_malformed_body():
    response = client.post("/chat", json={"not_messages": []})
    assert response.status_code == 422
