# 🧠 GeminiEventHandler

The `GeminiEventHandler` is a production-ready implementation of the [`AIAgentEventHandler`](https://github.com/ideabosque/ai_agent_handler) base class designed to interface with Google Gemini models. It provides robust message handling, model invocation, tool integration, streaming support, and automatic retry logic for reliable AI agent orchestration.

This handler enables a **stateless, multi-turn AI orchestration** system with support for:
- ✅ Function calling (tools/MCP servers)
- ✅ Streaming and non-streaming responses
- ✅ Automatic retry on empty responses (max 5 retries)
- ✅ Schema sanitization for MCP server compatibility
- ✅ File upload support (images, documents, etc.)
- ✅ Vertex AI and Google AI Studio support

---

## 🖉 Inheritance

![AI Agent Event Handler Class Diagram](/images/ai_agent_event_handler_class_diagram.jpg)

```
AIAgentEventHandler
     ▲
     └── GeminiEventHandler
```

---

## 📦 Key Features

### 🔧 Core Attributes

* **`client`**: Gemini API client instance (supports both Vertex AI and Google AI)
* **`model`**: Model name (e.g., `gemini-2.5-pro`, `gemini-2.0-flash-exp`)
* **`model_setting`**: Configuration dict containing `temperature`, `tools`, `system_instruction`, etc.
* **`final_output`**: Final response with `message_id`, `role`, and `content`

### 🛡️ Reliability Features

* **Schema Sanitization**: Automatically removes incompatible JSON Schema fields from MCP server tool definitions
* **Retry Logic**: Automatically retries empty responses up to 5 times before failing
* **Error Handling**: Comprehensive exception handling with proper cleanup
* **Content Validation**: Ensures `final_output` always has valid `message_id` and non-empty `content`

### 📞 Core Methods

#### `ask_model(input_messages, queue=None, stream_event=None, input_files=[], model_setting=None)`

Main entry point for making requests to Gemini.

**Args:**
- `input_messages`: List of conversation messages
- `queue`: Optional queue for streaming events
- `stream_event`: Optional threading.Event to signal completion
- `input_files`: Optional list of files to include
- `model_setting`: Optional settings to override defaults

**Returns:**
- `run_id` (non-streaming) or `None` (streaming)

**Features:**
- Automatic message processing and conversion
- File upload support
- Performance monitoring
- Streaming and non-streaming modes

#### `handle_response(response, input_messages, retry_count=0)`

Processes non-streaming responses with automatic retry logic.

**Scenarios:**
1. **Function call detected** → Execute function and recurse
2. **Empty response** → Retry up to 5 times
3. **Valid response** → Set `final_output`

#### `handle_stream(response_stream, input_messages, stream_event, retry_count=0)`

Processes streaming responses with real-time updates.

**Features:**
- Incremental text accumulation
- JSON/text format support
- Function call detection during streaming
- Automatic retry on empty streams
- WebSocket streaming support

#### `_sanitize_tool_schema(tool)`

Removes JSON Schema fields incompatible with Gemini API.

**Removes:**
- `additional_properties` / `additionalProperties` (all types)
- `required` field from non-object types (arrays, strings, etc.)

**Keeps:**
- `required` field for object types

This ensures MCP server tool definitions work correctly with Gemini.

---

## 🔌 Initialization

### Google AI Studio (API Key)

```python
from gemini_agent_handler import GeminiEventHandler

handler = GeminiEventHandler(
    logger=logger,
    agent=agent_config,
    endpoint_id="your-endpoint-id"
)
```

### Vertex AI (GCP)

```python
from gemini_agent_handler import GeminiEventHandler

handler = GeminiEventHandler(
    logger=logger,
    agent=agent_config,
    project="your-gcp-project",
    location="us-central1"
)
```

---

## 📘 Agent Configuration

### Basic Configuration

```python
agent_config = {
    "instructions": "You are a helpful AI assistant...",
    "llm": {"llm_name": "gemini"},
    "configuration": {
        "model": "gemini-2.5-pro",
        "api_key": "your-api-key",  # Or use Vertex AI with project/location
        "temperature": 0,
        "tools": []  # Tool definitions (auto-sanitized)
    },
    "num_of_messages": 30,
    "tool_call_role": "assistant"
}
```

### With MCP Servers

```python
agent_config = {
    "instructions": "You are a shopping assistant...",
    "llm": {"llm_name": "gemini"},
    "mcp_servers": [
        {
            "name": "shopify_mcp_server",
            "setting": {
                "base_url": "https://your-store.myshopify.com/api/mcp",
                "headers": {"Content-Type": "application/json"}
            }
        }
    ],
    "configuration": {
        "model": "gemini-2.5-pro",
        "api_key": "your-api-key",
        "temperature": 0,
        "tools": []  # Tools auto-populated from MCP servers
    },
    "num_of_messages": 30,
    "tool_call_role": "assistant"
}
```

**Note:** Tools from MCP servers are automatically sanitized to remove:
- `additional_properties` / `additionalProperties`
- `required` fields from non-object types

This ensures compatibility with Gemini's schema validation.

### With Google Search

```python
agent_config = {
    "configuration": {
        "model": "gemini-2.5-pro",
        "api_key": "your-api-key",
        "tools": [{"name": "google_search"}]
    }
}
```

### With Code Execution

```python
agent_config = {
    "configuration": {
        "model": "gemini-2.5-pro",
        "api_key": "your-api-key",
        "tools": [{"name": "code_execution"}]
    }
}
```

---

## 🔄 Retry Logic

The handler automatically retries empty responses up to 5 times:

```python
# Automatic retry happens internally
handler.ask_model(input_messages)

# Logs on retry:
# "Received empty response from model, retrying (attempt 1/5)..."
# "Received empty response from model, retrying (attempt 2/5)..."
# ...

# After 5 retries:
# Exception: Maximum retry limit (5) exceeded for empty responses
```

**Retry scenarios:**
- ✅ Empty text response (no content)
- ✅ Whitespace-only response
- ✅ None response
- ✅ Empty streaming response

**No retry needed:**
- ✅ Function calls (handled normally)
- ✅ Valid text responses

---

## 📎 Gemini-Specific Notes

* **API Modes**: Supports both `generate_content` (non-stream) and `generate_content_stream` (streaming)
* **System Instructions**: Uses `system_instruction` field (automatically set from `instructions`)
* **Tool Formats**: Supports function declarations, Google Search, and code execution
* **File Support**: Upload images, PDFs, documents via `insert_file()`
* **Vertex AI**: Set `project` and `location` instead of `api_key`
* **Streaming**: Real-time chunk processing with WebSocket support

---
## 💬 Usage Examples

### Non-Streaming Request

```python
from gemini_agent_handler import GeminiEventHandler
import logging

logger = logging.getLogger(__name__)
handler = GeminiEventHandler(logger, agent_config, endpoint_id="test")

# Simple request
input_messages = [
    {"role": "user", "content": "What is the weather in Tokyo?"}
]
run_id = handler.ask_model(input_messages)

# Access response
print(handler.final_output)
# {
#   "message_id": "msg-gemini-gemini-2.5-pro-1234567890-abcd1234",
#   "role": "assistant",
#   "content": "I'll check the weather in Tokyo for you..."
# }
```

### Streaming Request

```python
import threading
from queue import Queue

queue = Queue()
event = threading.Event()

# Start streaming request
thread = threading.Thread(
    target=handler.ask_model,
    args=[input_messages, queue, event],
    daemon=True
)
thread.start()

# Get run_id
run_id = queue.get()["value"]

# Wait for completion
event.wait()
print(handler.final_output["content"])
```

### With File Upload

```python
import base64

# Read and encode file
with open("document.pdf", "rb") as f:
    encoded_content = base64.b64encode(f.read()).decode("utf-8")

# Upload file
input_files = [{
    "filename": "document.pdf",
    "encoded_content": encoded_content,
    "mime_type": "application/pdf"
}]

# Send message with file
input_messages = [
    {"role": "user", "content": "Summarize this document"}
]
handler.ask_model(input_messages, input_files=input_files)
```

---

## 🛠️ Troubleshooting

### Schema Validation Errors

**Problem:**
```
ValidationError: Extra inputs are not permitted [type=extra_forbidden]
```

**Solution:** Tools are auto-sanitized. If you see this error, ensure you're using the latest version.

---

### Empty Response Retry

**Problem:**
```
Maximum retry limit (5) exceeded for empty responses
```

**Solutions:**
1. Check your prompt/instructions aren't causing the model to refuse
2. Verify your API key/credentials are valid
3. Check if the model is overloaded (try different model)
4. Review model settings (temperature, max_tokens, etc.)

---

### Function Call Loop

**Problem:** Model keeps calling same function repeatedly

**Solutions:**
1. Ensure function returns meaningful data
2. Check function description is clear
3. Verify function response format is correct
4. Review conversation history for loops

---

### MCP Server Tools Not Working

**Problem:** Tools from MCP servers cause validation errors

**Solution:** Schema sanitization is automatic. Check:
1. MCP server returns valid JSON Schema
2. Tool names don't conflict with built-in tools
3. MCP server is accessible from handler

---

## 📊 Performance Monitoring

The handler includes built-in performance monitoring via `@Utility.performance_monitor.monitor_operation`:

```python
# Logs automatically generated:
# "Gemini: ask_model completed in 2.5s"
# "Gemini: ask_model failed after 1.2s: <error>"
```

---

## 🔒 Security Notes

- **API Keys**: Never commit API keys to version control
- **Function Execution**: Functions run in handler context - validate inputs
- **File Uploads**: Validate file types and sizes before upload
- **MCP Servers**: Only connect to trusted MCP servers

---

## 📦 Installation

### From PyPI

```bash
pip install gemini-agent-handler
```

### With Optional Dependencies

```bash
# Development tools (black, mypy, pytest, etc.)
pip install gemini-agent-handler[dev]

# Testing tools
pip install gemini-agent-handler[test]

# Vertex AI support
pip install gemini-agent-handler[vertex]

# Multiple groups
pip install gemini-agent-handler[dev,vertex]
```

### From Source

```bash
git clone https://github.com/ideabosque/gemini_agent_handler.git
cd gemini_agent_handler
pip install -e ".[dev]"
```

---

## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/ideabosque/gemini_agent_handler.git
cd gemini_agent_handler

# Install with development dependencies
pip install -e ".[dev]"
```

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest gemini_agent_handler/tests/test_gemini_agent_handler.py

# Run with coverage
pytest --cov=gemini_agent_handler --cov-report=html
```

### Code Quality

```bash
# Format code
black gemini_agent_handler/

# Type checking
mypy gemini_agent_handler/

# Linting
flake8 gemini_agent_handler/
```

### Build Package

```bash
# Install build tool
pip install build

# Build wheel and source distribution
python -m build

# Output: dist/gemini_agent_handler-0.0.1-py3-none-any.whl
#         dist/gemini_agent_handler-0.0.1.tar.gz
```

---

## 📚 Related Documentation

- [AIAgentEventHandler Base Class](https://github.com/ideabosque/ai_agent_handler)
- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [MCP Protocol Specification](https://modelcontextprotocol.io)

---

## 📝 License

See LICENSE file for details.

---