from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data

def test_ingest_endpoint():
    with patch("app.main.ingest_data") as mock_ingest:
        response = client.post("/ingest")
        assert response.status_code == 200
        assert "Ingestion task started" in response.json()["message"]

@patch("app.main.retrieve_documents")
@patch("app.main.generate_rag_response", new_callable=AsyncMock)
def test_query_endpoint(mock_generate, mock_retrieve):
    # Mock the retrieved subset from Endee DB
    mock_retrieve.return_value = [{"id": "test_id_1", "score": 0.95, "content": "{'category': 'test'}", "filter": ""}]
    
    # Mock LLM API generation call
    mock_generate.return_value = "This is a mocked LLM answer."

    payload = {
        "query": "What is the latest test threat?",
        "top_k": 2,
        "stream": False
    }
    
    response = client.post("/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["answer"] == "This is a mocked LLM answer."
    assert len(data["retrieved_documents"]) == 1
    assert data["retrieved_documents"][0]["id"] == "test_id_1"
    
    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once()
