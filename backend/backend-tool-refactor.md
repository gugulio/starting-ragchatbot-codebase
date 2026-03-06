Refactor @backend/ai_generator.py to support sequential tool calling where Claude can
make up to 2 tool calls in separate API rounds.

# 1. 清晰定義現狀

Current behavior:
- Claude makes 1 tool call -> tools are removed from API params -> final response
- If Claude wants another tool call after seeing results, it can't(gets empty response)

# 2. 清晰定義目標

Desired behavior:
- Each tool call should be a separate API request where Claude can reason about
previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions,
or when information from different courses/lessons is needed 

# 3. 提供具體示例

Example flow:
1. User:"Search for a course that discusses the same topic as lesson 4 of course X"
2. Claude: get course outline for course X - gets title of lesson 4
3. Claude: uses the title to search for a course that discusses the same
topic -> returns course information
4. Claude: provides complete answer

# 4. 給出明確的技術約束

Requirements:
- Maximum 2 sequential rounds per user query
- Terminate when:(a)2 rounds completed,(b)Claude's response has no tool_use blocks,
or (c)tool call fails
- Preserve conversation context between rounds
- Handle tool execution errors gracefully

Notes:
- Updates the system prompt in @backend/ai_generator.py
- Updates the test @backend/tests/test_ai_generator.py
- Write tests that verify the external behavior (API calls made, tools executed, results returned)
rather than internal state details. 

# 5. 提出元指令:派出子智能体

Use two parallel subagents to brainstorm possible plans. Do not implement any code.























