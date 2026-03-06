# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

**Prerequisites:** Python 3.13+, `uv`, and a NVIDIA NIM API key in `.env`:
```
NVIDIA_API_KEY=nvapi-...
```
Get a free key (1000 credits on sign-up) at https://build.nvidia.com.

```bash
# Quick start
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

`uv run` installs all dependencies automatically from `pyproject.toml` — no separate install step needed.

## Architecture

This is a single-process full-stack app. FastAPI serves both the REST API and the static frontend from the same Uvicorn server. All backend modules live flat in `backend/` with no sub-packages.

### Request flow for a user query

1. `frontend/script.js` — `sendMessage()` POSTs `{ query, session_id, model }` to `/api/query`
2. `backend/app.py` — creates a session if needed, delegates to `RAGSystem.query()`
3. `backend/rag_system.py` — fetches conversation history, calls `AIGenerator.generate_response()` with tool definitions
4. **First NIM API call** — model decides whether to answer directly or call `search_course_content`
5. If tool called → `search_tools.py` → `vector_store.py` → ChromaDB semantic search → returns top-5 chunks
6. **Second NIM API call** — model generates a final answer from the retrieved chunks
7. Response + sources bubble back up; `session_manager.py` stores the exchange

### Key design constraints

- **One search per query**: Claude is instructed to call `search_course_content` at most once per turn.
- **Session history is in-memory only**: Sessions are lost on server restart. `MAX_HISTORY = 2` means only the last 2 exchanges are sent to Claude.
- **ChromaDB is persistent**: Data survives restarts at `backend/chroma_db/`. On startup, already-indexed courses are skipped (matched by title).
- **Tool calling uses two API calls**: The first call may trigger a tool; the second call (without tools) generates the final answer.

### Adding a new course document

Drop a `.txt` file into `docs/` following this exact format:
```
Course Title: My Course Name
Course Link: https://example.com/course
Course Instructor: Instructor Name

Lesson 1: Lesson Title
Lesson Link: https://example.com/lesson1
... lesson content ...

Lesson 2: Another Lesson
... lesson content ...
```

The file is auto-ingested on next server startup. The course title is used as the unique ID in ChromaDB — renaming a title creates a duplicate.

### Configuration

All tuneable values are in `backend/config.py` (single `Config` dataclass, instantiated as `config`):

| Key | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 800 | Max characters per vector chunk |
| `CHUNK_OVERLAP` | 100 | Character overlap between chunks |
| `MAX_RESULTS` | 5 | Top-k chunks returned by vector search |
| `MAX_HISTORY` | 2 | Conversation turns kept in session |
| `DEFAULT_MODEL` | `meta/llama-3.1-8b-instruct` | Default NVIDIA NIM model |
| `AVAILABLE_MODELS` | (list) | Models shown in the UI dropdown |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB persistence path (relative to `backend/`) |

### Extending the tool system

`search_tools.py` uses a simple plugin pattern. To add a new tool:
1. Subclass `Tool` (abstract base in `search_tools.py`) and implement `get_tool_definition()` and `execute()`
2. Register it in `rag_system.py`: `self.tool_manager.register_tool(MyNewTool(...))`

The tool definition dict must follow the OpenAI tool schema (`{ "type": "function", "function": { ... } }`). The `ToolManager` passes all registered tool definitions to every API call.
