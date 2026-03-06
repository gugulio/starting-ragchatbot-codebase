import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

import rag_system as rag_system_mod
import ai_generator as ai_generator_mod


def make_mock_response(finish_reason="stop", content="Answer", tool_calls=None):
    """Create a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [choice]
    return response


@pytest.fixture
def mock_openai_client():
    return MagicMock()


@pytest.fixture
def rag_system(tmp_path, monkeypatch, mock_openai_client):
    # Avoid loading SentenceTransformer / ChromaDB during tests
    mock_store = MagicMock()
    monkeypatch.setattr(rag_system_mod, "VectorStore", MagicMock(return_value=mock_store))
    monkeypatch.setattr(rag_system_mod, "DocumentProcessor", MagicMock())

    # Use real AIGenerator but with a mocked HTTP client so Fix A is exercised
    monkeypatch.setattr(ai_generator_mod, "OpenAI", MagicMock(return_value=mock_openai_client))

    config = SimpleNamespace(
        CHROMA_PATH=str(tmp_path / "chroma"),
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        MAX_RESULTS=5,
        MAX_HISTORY=2,
        DEFAULT_MODEL="test-model",
        NVIDIA_API_KEY="test-key",
        NVIDIA_BASE_URL="https://example.com",
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
    )

    from rag_system import RAGSystem
    system = RAGSystem(config)
    return system


def test_query_returns_string_and_list(rag_system, mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = make_mock_response(content="answer")

    answer, sources = rag_system.query("What is RAG?", session_id="s1")

    assert isinstance(answer, str)
    assert isinstance(sources, list)


def test_query_does_not_return_none_answer(rag_system, mock_openai_client):
    """Bug A: when the API returns content=None, query() must not return None as the answer."""
    mock_openai_client.chat.completions.create.return_value = make_mock_response(
        finish_reason="stop", content=None
    )

    answer, sources = rag_system.query("What is RAG?", session_id="s1")

    assert answer is not None, "query() must never return None as the answer"
    assert isinstance(answer, str)


def test_session_history_updated_after_query(rag_system, mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = make_mock_response(content="answer")
    session_id = "s_history"

    rag_system.query("First question", session_id=session_id)
    rag_system.query("Second question", session_id=session_id)

    history = rag_system.session_manager.get_conversation_history(session_id)
    assert history is not None
    assert "First question" in history
    assert "Second question" in history


def test_sources_reset_between_queries(rag_system, mock_openai_client):
    mock_openai_client.chat.completions.create.return_value = make_mock_response(content="answer")
    session_id = "s_sources"

    # Simulate that the first query produced sources via a prior tool call
    rag_system.search_tool.last_sources = [{"label": "Course 1 - Lesson 1", "url": None}]

    answer1, sources1 = rag_system.query("First question", session_id=session_id)
    assert len(sources1) == 1

    # Second query: sources must be empty (reset after the first query)
    answer2, sources2 = rag_system.query("Second question", session_id=session_id)
    assert len(sources2) == 0


def test_exception_from_ai_propagates(rag_system, mock_openai_client):
    """Exceptions from the AI layer should propagate so app.py can return HTTP 500."""
    mock_openai_client.chat.completions.create.side_effect = RuntimeError("API Error")

    with pytest.raises(RuntimeError, match="API Error"):
        rag_system.query("What is RAG?", session_id="s1")
