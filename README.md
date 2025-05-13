# 🧠 GeminiEventHandler

The `GeminiEventHandler` is a concrete implementation of the `AIAgentEventHandler` base class designed to interface with Google Gemini models. It orchestrates message formatting, model invocation, tool integration, streaming, and threading within the AI agent execution pipeline.

This handler enables a **stateless, multi-turn AI orchestration** system built to support tools like `get_weather_forecast`.

---

## 🖉 Inheritance

![AI Agent Event Handler Class Diagram](/images/ai_agent_event_handler_class_diagram.jpg)

```
AIAgentEventHandler
     ▲
     └── GeminiEventHandler
```

---

## 📦 Module Features

### 🔧 Attributes

* `client`: Gemini API client instance
* `model_settings`: A dictionary containing Gemini model configuration (e.g., `model`, `temperature`, `tools`, etc.)

### 📞 Core Method: `invoke_model`

```python
def invoke_model(self, **kwargs: Dict[str, Any]) -> Any:
    """
    Invokes the Gemini model with the provided configuration.

    Args:
        kwargs: Dictionary containing:
            - input: Messages to send to the model
            - stream: Boolean indicating if streaming response is desired

    Returns:
        Either a streaming or non-streaming model response

    Raises:
        Exception: If model invocation fails
    """
    try:
        config = types.GenerateContentConfig(**self.model_setting)

        if kwargs.get("stream"):
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=kwargs["input"],
                config=config,
            )

        return self.client.models.generate_content(
            model=self.model,
            contents=kwargs["input"],
            config=config,
        )
    except Exception as e:
        self.logger.error(f"Error invoking model: {str(e)}")
        raise Exception(f"Failed to invoke model: {str(e)}")
```

---

## 📘 Sample Configuration (Gemini)

```json
{
  "endpoint_id": "google",
  "agent_name": "Weather Assistant (Gemini)",
  "model": "gemini-2.5-pro-preview-03-25",
  "temperature": 0,
  "tools": [
    {
      "type": "function",
      "name": "get_weather_forecast",
      "description": "Get the weather forecast for a given city and date",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The city to retrieve the forecast for."
          },
          "date": {
            "type": "string",
            "description": "The forecast date (YYYY-MM-DD)."
          }
        },
        "required": ["city", "date"],
        "additionalProperties": false
      }
    }
  ],
  "functions": {
    "get_weather_forecast": {
      "class_name": "WeatherForecastFunction",
      "module_name": "weather_funct",
      "configuration": {}
    }
  },
  "function_configuration": {
    "weather_provider": "open-meteo",
    "region_name": "${region_name}",
    "aws_access_key_id": "${aws_access_key_id}",
    "aws_secret_access_key": "${aws_secret_access_key}"
  },
  "instructions": "You are a Gemini-based AI Assistant responsible for providing accurate weather information using the `get_weather_forecast` function. Analyze user input to extract city and date information, and call the tool accordingly. Always clarify ambiguous input and offer detailed yet concise responses.",
  "num_of_messages": 30,
  "tool_call_role": "assistant",
  "api_key": "${gemini_api_key}"
}
```

---

## 📎 Gemini-Specific Notes

* Gemini API uses `generate_content` (non-stream) and `generate_content_stream` (streaming).
* Use `system_instruction` instead of `instructions`.
* Streaming output is handled via a server-side generator.

---
## 💬 Full-Scale Chatbot Scripts

### 🔁 Non-Streaming Chatbot Script

```python
import logging
import os
import sys

import pendulum
from dotenv import load_dotenv
from ai_agent_handler import AIAgentEventHandler
from gemini_agent_handler import GeminiEventHandler

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "funct_bucket_name": os.getenv("funct_bucket_name"),
    "funct_zip_path": os.getenv("funct_zip_path"),
    "funct_extract_path": os.getenv("funct_extract_path"),
    "connection_id": os.getenv("connection_id"),
    "endpoint_id": os.getenv("endpoint_id"),
    "test_mode": os.getenv("test_mode"),
}

weather_agent = { ... }  # Configuration as defined above
handler = GeminiEventHandler(logger=None, agent=weather_agent, **setting)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])
    run_id = handler.ask_model(messages)

    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

### 🔁 Streaming Chatbot Script

```python
import logging
import os
import sys

import pendulum
from dotenv import load_dotenv
from ai_agent_handler import AIAgentEventHandler
from gemini_agent_handler import GeminiEventHandler

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

load_dotenv()
setting = {
    "region_name": os.getenv("region_name"),
    "aws_access_key_id": os.getenv("aws_access_key_id"),
    "aws_secret_access_key": os.getenv("aws_secret_access_key"),
    "funct_bucket_name": os.getenv("funct_bucket_name"),
    "funct_zip_path": os.getenv("funct_zip_path"),
    "funct_extract_path": os.getenv("funct_extract_path"),
    "connection_id": os.getenv("connection_id"),
    "endpoint_id": os.getenv("endpoint_id"),
    "test_mode": os.getenv("test_mode"),
}

weather_agent = { ... }  # Configuration as defined above
handler = GeminiEventHandler(logger=None, agent=weather_agent, **setting)
handler.short_term_memory = []

def get_input_messages(messages, num_of_messages):
    return [msg["message"] for msg in sorted(messages, key=lambda x: x["created_at"], reverse=True)][:num_of_messages][::-1]

while True:
    user_input = input("User: ")
    if user_input.strip().lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break

    message = {"role": "user", "content": user_input}
    handler.short_term_memory.append({"message": message, "created_at": pendulum.now("UTC")})
    messages = get_input_messages(handler.short_term_memory, weather_agent["num_of_messages"])

    stream_queue = Queue()
    stream_event = threading.Event()
    stream_thread = threading.Thread(
        target=handler.ask_model,
        args=[messages, stream_queue, stream_event],
        daemon=True
    )
    stream_thread.start()

    result = stream_queue.get()
    if result["name"] == "run_id":
        print("Run ID:", result["value"])

    stream_event.wait()
    print("Chatbot:", handler.final_output["content"])
    handler.short_term_memory.append({
        "message": handler.final_output,
        "created_at": pendulum.now("UTC")
    })
```

---