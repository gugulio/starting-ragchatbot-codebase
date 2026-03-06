import pytest
from unittest.mock import MagicMock, patch

from ai_generator import AIGenerator


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
def mock_client():
    return MagicMock()


@pytest.fixture
def generator(mock_client):
    with patch("ai_generator.OpenAI", return_value=mock_client):
        gen = AIGenerator(
            api_key="test-key",
            base_url="https://example.com",
            default_model="test-model",
        )
    return gen, mock_client


def test_direct_response_no_tools(generator):
    gen, client = generator
    client.chat.completions.create.return_value = make_mock_response(
        finish_reason="stop", content="Hello"
    )

    result = gen.generate_response(query="Hi")

    assert result == "Hello"


def test_none_content_does_not_return_none(generator):
    """Bug A: when the model returns content=None, generate_response must not return None."""
    gen, client = generator
    client.chat.completions.create.return_value = make_mock_response(
        finish_reason="stop", content=None
    )

    result = gen.generate_response(query="Hi")

    assert result is not None, "generate_response must never return None"
    assert isinstance(result, str)


def test_tool_call_triggers_second_api_call(generator):
    gen, client = generator
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "RAG"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call])
    second = make_mock_response(finish_reason="stop", content="Final answer")
    client.chat.completions.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    result = gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert result == "Final answer"
    assert client.chat.completions.create.call_count == 2


def test_tool_manager_executed_with_correct_args(generator):
    gen, client = generator
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "RAG"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call])
    second = make_mock_response(finish_reason="stop", content="Final answer")
    client.chat.completions.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    tool_manager.execute_tool.assert_called_once_with("search_course_content", query="RAG")


def test_tool_result_appended_to_messages(generator):
    gen, client = generator
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "RAG"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call])
    second = make_mock_response(finish_reason="stop", content="Final answer")
    client.chat.completions.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "tool result content"

    gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "tool result content"
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_forced_final_call_has_no_tools_param(generator):
    """In a 2-round scenario, the 3rd (forced final) call must not include tools."""
    gen, client = generator
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_1"
    tool_call_1.function.name = "search_course_content"
    tool_call_1.function.arguments = '{"query": "RAG"}'

    tool_call_2 = MagicMock()
    tool_call_2.id = "call_2"
    tool_call_2.function.name = "search_course_content"
    tool_call_2.function.arguments = '{"query": "transformers"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_1])
    second = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_2])
    third = make_mock_response(finish_reason="stop", content="Final answer")
    client.chat.completions.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    third_call_kwargs = client.chat.completions.create.call_args_list[2].kwargs
    assert "tools" not in third_call_kwargs


# --- New tests for sequential tool calling ---

def test_two_round_success(generator):
    """Two full tool rounds complete; result comes from the forced 3rd call."""
    gen, client = generator
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_1"
    tool_call_1.function.name = "search_course_content"
    tool_call_1.function.arguments = '{"query": "RAG"}'

    tool_call_2 = MagicMock()
    tool_call_2.id = "call_2"
    tool_call_2.function.name = "search_course_content"
    tool_call_2.function.arguments = '{"query": "transformers"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_1])
    second = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_2])
    third = make_mock_response(finish_reason="stop", content="Final answer from round 2")
    client.chat.completions.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    result = gen.generate_response(
        query="Find courses on two topics",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert result == "Final answer from round 2"
    assert client.chat.completions.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2


def test_one_round_model_stops_naturally(generator):
    """After one tool round the model answers directly; only 2 API calls are made."""
    gen, client = generator
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "RAG"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call])
    second = make_mock_response(finish_reason="stop", content="Natural answer")
    client.chat.completions.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    result = gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert result == "Natural answer"
    assert client.chat.completions.create.call_count == 2


def test_round_limit_forced_termination(generator):
    """MAX_TOOL_ROUNDS exhausted triggers a forced final call; result is from that 3rd call."""
    gen, client = generator
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_1"
    tool_call_1.function.name = "search_course_content"
    tool_call_1.function.arguments = '{"query": "RAG"}'

    tool_call_2 = MagicMock()
    tool_call_2.id = "call_2"
    tool_call_2.function.name = "search_course_content"
    tool_call_2.function.arguments = '{"query": "transformers"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_1])
    second = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_2])
    third = make_mock_response(finish_reason="stop", content="Forced final answer")
    client.chat.completions.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    result = gen.generate_response(
        query="Complex query",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert result == "Forced final answer"
    assert client.chat.completions.create.call_count == 3
    third_call_kwargs = client.chat.completions.create.call_args_list[2].kwargs
    assert "tools" not in third_call_kwargs


def test_tool_error_still_produces_answer(generator):
    """A tool execution error is appended as a tool message; the loop continues to a 2nd call."""
    gen, client = generator
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "search_course_content"
    tool_call.function.arguments = '{"query": "RAG"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call])
    second = make_mock_response(finish_reason="stop", content="Answer after error")
    client.chat.completions.create.side_effect = [first, second]

    tool_manager = MagicMock()
    tool_manager.execute_tool.side_effect = RuntimeError("DB connection failed")

    result = gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert result == "Answer after error"
    assert client.chat.completions.create.call_count == 2
    second_call_kwargs = client.chat.completions.create.call_args_list[1].kwargs
    messages = second_call_kwargs["messages"]
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Tool execution failed" in tool_msgs[0]["content"]


def test_messages_accumulate_across_rounds(generator):
    """After 2 tool rounds the forced final call receives all 6 messages."""
    gen, client = generator
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_1"
    tool_call_1.function.name = "search_course_content"
    tool_call_1.function.arguments = '{"query": "RAG"}'

    tool_call_2 = MagicMock()
    tool_call_2.id = "call_2"
    tool_call_2.function.name = "search_course_content"
    tool_call_2.function.arguments = '{"query": "transformers"}'

    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_1])
    second = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_2])
    third = make_mock_response(finish_reason="stop", content="Final answer")
    client.chat.completions.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    gen.generate_response(
        query="What is RAG?",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    third_call_kwargs = client.chat.completions.create.call_args_list[2].kwargs
    messages = third_call_kwargs["messages"]
    assert len(messages) == 6  # system, user, asst×2, tool×2
    roles = [m["role"] for m in messages]
    assert roles.count("assistant") == 2
    assert roles.count("tool") == 2


def test_cannot_exceed_max_tool_rounds(generator):
    """With 3 tool-call responses available, only 3 API calls happen (not 4)."""
    gen, client = generator
    tool_call_1 = MagicMock()
    tool_call_1.id = "call_1"
    tool_call_1.function.name = "search_course_content"
    tool_call_1.function.arguments = '{"query": "RAG"}'

    tool_call_2 = MagicMock()
    tool_call_2.id = "call_2"
    tool_call_2.function.name = "search_course_content"
    tool_call_2.function.arguments = '{"query": "transformers"}'

    tool_call_3 = MagicMock()
    tool_call_3.id = "call_3"
    tool_call_3.function.name = "search_course_content"
    tool_call_3.function.arguments = '{"query": "attention"}'

    # 3 tool-call responses; the 3rd is the forced-final slot — MAX_TOOL_ROUNDS prevents a 4th call
    first = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_1])
    second = make_mock_response(finish_reason="tool_calls", content=None, tool_calls=[tool_call_2])
    third = make_mock_response(finish_reason="tool_calls", content="Forced final", tool_calls=[tool_call_3])
    client.chat.completions.create.side_effect = [first, second, third]

    tool_manager = MagicMock()
    tool_manager.execute_tool.return_value = "search results"

    gen.generate_response(
        query="Complex query",
        tools=[{"type": "function"}],
        tool_manager=tool_manager,
    )

    assert client.chat.completions.create.call_count == 3
    assert tool_manager.execute_tool.call_count == 2
