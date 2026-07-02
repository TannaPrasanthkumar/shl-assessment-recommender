"""
API router integration tests verifying schema compliance and routing actions.
Uses standard TestClient to assert response formatting conventions.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client() -> TestClient:
    """
    Provides a FastAPI test client instance, running startup events under lifespan context.
    """
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client: TestClient) -> None:
    """
    Verifies the /health readiness probe.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_chat_stateless_post_valid(client: TestClient) -> None:
    """
    Verifies that POST /chat parses a valid conversational message history.
    """
    payload = {
        "messages": [
            {"role": "user", "content": "I am looking for a Java Developer assessment."},
            {"role": "assistant", "content": "Sure, what is the seniority level?"},
            {"role": "user", "content": "Senior role with 8 years of experience."}
        ]
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data
    assert isinstance(data["recommendations"], list)
    assert isinstance(data["end_of_conversation"], bool)


def test_chat_empty_messages_validation_error(client: TestClient) -> None:
    """
    Verifies that sending an empty messages array triggers schema validation error.
    """
    payload = {"messages": []}
    response = client.post("/chat", json=payload)
    assert response.status_code == 422  # Unprocessable Entity
