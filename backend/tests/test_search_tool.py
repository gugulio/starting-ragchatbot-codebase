import pytest
from unittest.mock import MagicMock

from search_tools import CourseSearchTool
from vector_store import SearchResults


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def tool(mock_store):
    return CourseSearchTool(mock_store)


def test_basic_search_returns_formatted_text(tool, mock_store):
    mock_store.search.return_value = SearchResults(
        documents=["Content A", "Content B"],
        metadata=[
            {"course_title": "Course 1", "lesson_number": 1},
            {"course_title": "Course 2", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )
    mock_store.get_lesson_link.return_value = None

    result = tool.execute(query="RAG")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "Content A" in result


def test_empty_results_returns_no_content_message(tool, mock_store):
    mock_store.search.return_value = SearchResults([], [], [])

    result = tool.execute(query="RAG")

    assert "No relevant content found" in result


def test_error_result_returns_error_string(tool, mock_store):
    mock_store.search.return_value = SearchResults.empty("Search error: connection failed")

    result = tool.execute(query="RAG")

    assert "Search error: connection failed" in result


def test_sources_stored_in_last_sources(tool, mock_store):
    mock_store.search.return_value = SearchResults(
        documents=["Content A"],
        metadata=[{"course_title": "Course 1", "lesson_number": 1}],
        distances=[0.1],
    )
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"

    tool.execute(query="RAG")

    assert len(tool.last_sources) == 1


def test_source_deduplication(tool, mock_store):
    mock_store.search.return_value = SearchResults(
        documents=["Content A", "Content B"],
        metadata=[
            {"course_title": "Course 1", "lesson_number": 1},
            {"course_title": "Course 1", "lesson_number": 1},
        ],
        distances=[0.1, 0.2],
    )
    mock_store.get_lesson_link.return_value = None

    tool.execute(query="RAG")

    assert len(tool.last_sources) == 1


def test_course_name_forwarded_to_store(tool, mock_store):
    mock_store.search.return_value = SearchResults([], [], [])

    tool.execute(query="RAG", course_name="MCP")

    mock_store.search.assert_called_once_with(
        query="RAG", course_name="MCP", lesson_number=None
    )


def test_lesson_number_forwarded_to_store(tool, mock_store):
    mock_store.search.return_value = SearchResults([], [], [])

    tool.execute(query="RAG", lesson_number=3)

    mock_store.search.assert_called_once_with(
        query="RAG", course_name=None, lesson_number=3
    )
