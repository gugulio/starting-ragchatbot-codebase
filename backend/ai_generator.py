import json
from openai import OpenAI
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with NVIDIA NIM API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Two sequential searches maximum per query** — use a second search only when the first result is
  insufficient or the query explicitly requires information from two different sources
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, base_url: str, default_model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model

    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         model: Optional[str] = None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            model: Optional model override

        Returns:
            Generated response as string
        """
        active_model = model or self.default_model

        # Build system message content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        api_params: Dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**api_params)
        tool_execution_count = 0

        while response.choices[0].message.tool_calls and tool_manager:
            assistant_message = response.choices[0].message

            # Append clean assistant tool-call message (content=None avoids raw markup echo)
            clean_assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
            messages = messages + [clean_assistant_msg]

            # Execute each tool call; on error append error result and stop the for loop
            for tool_call in assistant_message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    tool_result = tool_manager.execute_tool(tool_call.function.name, **arguments)
                except Exception as e:
                    tool_result = f"Tool execution failed: {e}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                    break
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

            tool_execution_count += 1

            if tool_execution_count >= self.MAX_TOOL_ROUNDS:
                # Rounds exhausted — force a final answer without tools
                final_response = self.client.chat.completions.create(
                    model=active_model,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                )
                return final_response.choices[0].message.content or ""

            # Intermediate call WITH tools — model may answer directly or call another tool
            next_api_params: Dict[str, Any] = {
                "model": active_model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 800,
            }
            if tools:
                next_api_params["tools"] = tools
                next_api_params["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**next_api_params)

        # Model answered directly (no tool_calls in latest response)
        return response.choices[0].message.content or ""
