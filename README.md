# 🤖 AI Agent Toolkit

A framework-agnostic toolkit for building AI agents with tools, memory, and multi-step reasoning. Built for production, designed for extensibility.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-0.1.0-orange)

---

## ✨ Features

- 🧠 **Conversational Memory** - Persistent conversation history with multiple strategies
- 🔧 **Tool System** - Register and call functions with automatic schema generation
- 🧠 **ReAct Agent** - Reasoning + Acting for complex multi-step tasks
- 📡 **Streaming Support** - Real-time token-by-token responses
- 🔄 **Retry Logic** - Built-in error handling and retries
- 📊 **Observability** - Detailed logging and tracing

---

## 📦 Installation

```bash
pip install ai-agent-toolkit
```

Or install from source:

```bash
git clone https://github.com/Moses-main/ai-agent-toolkit.git
cd ai-agent-toolkit
pip install -e .
```

---

## 🚀 Quick Start

```python
from ai_agent import Agent, OpenAIClient
from ai_agent.tools import calculator, search

# Initialize the client
client = OpenAIClient(api_key="your-api-key")

# Create an agent with tools
agent = Agent(
    client=client,
    model="gpt-4",
    tools=[calculator, search],
    system_prompt="You are a helpful assistant with access to tools."
)

# Run a simple query
response = agent.run("What is 15 + 27 multiplied by 3?")
print(response)
```

---

## 🏗️ Architecture

```
ai-agent-toolkit/
├── ai_agent/
│   ├── __init__.py
│   ├── agent.py          # Core agent logic
│   ├── client.py         # LLM client interface
│   ├── memory.py         # Memory management
│   ├── tools/            # Tool system
│   │   ├── __init__.py
│   │   ├── base.py       # Base tool class
│   │   ├── registry.py   # Tool registry
│   │   └── builtins.py   # Built-in tools
│   └── types.py          # Type definitions
├── examples/
│   ├── simple_agent.py
│   ├── react_agent.py
│   └── memory_demo.py
└── tests/
```

---

## 💻 Usage Examples

### 1. Simple Agent

```python
from ai_agent import Agent, OpenAIClient

client = OpenAIClient(api_key="your-key")
agent = Agent(client=client, model="gpt-4")

response = agent.run("Hello, how are you?")
print(response)
```

### 2. Agent with Tools

```python
from ai_agent import Agent, OpenAIClient
from ai_agent.tools import calculator, date_tool

# Define custom tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    # Your weather API logic here
    return f"The weather in {location} is sunny, 72°F"

agent = Agent(
    client=client,
    tools=[calculator, date_tool, get_weather]
)

response = agent.run("What's the weather in Lagos and what's 100 * 25?")
print(response)
```

### 3. ReAct Agent (Reasoning + Acting)

```python
from ai_agent import ReActAgent, OpenAIClient

agent = ReActAgent(
    client=client,
    model="gpt-4",
    max_iterations=10
)

# The agent will think, act, and observe iteratively
result = agent.run("""
    I need to buy 3 items that cost $15.99 each.
    There's a 10% tax. How much total?
""")
print(result.final_answer)
```

### 4. Agent with Memory

```python
from ai_agent import Agent, OpenAIClient
from ai_agent.memory import ConversationBufferMemory

# Create memory with history
memory = ConversationBufferMemory()

agent = Agent(
    client=client,
    memory=memory,
    system_prompt="You remember previous conversations."
)

# First interaction
agent.run("My name is Moses.")

# Second interaction - agent remembers!
response = agent.run("What's my name?")
# Response: "Your name is Moses."
```

### 5. Streaming Responses

```python
from ai_agent import Agent, OpenAIClient

agent = Agent(client=client, model="gpt-4")

# Stream tokens in real-time
for token in agent.run_stream("Write a story about AI"):
    print(token, end="", flush=True)
```

---

## 🔧 Creating Custom Tools

### Basic Tool

```python
from ai_agent.tools import Tool, tool

# Using decorator
@tool(name="multiply", description="Multiply two numbers")
def multiply(a: float, b: float) -> float:
    """Multiply a by b."""
    return a * b

# Or extend base class
from ai_agent.tools.base import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "Does something useful"
    
    def execute(self, **kwargs):
        # Your logic here
        return "result"
```

### Tool with Complex Return

```python
from ai_agent.tools import tool

@tool(
    name="search_products",
    description="Search for products in inventory",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "category": {"type": "string"},
            "max_results": {"type": "integer", "default": 10}
        },
        "required": ["query"]
    }
)
def search_products(query: str, category: str = None, max_results: int = 10):
    """Search products with filters."""
    # Your search logic
    return [
        {"name": "Product 1", "price": 29.99},
        {"name": "Product 2", "price": 49.99},
    ]
```

---

## 🔌 Supported LLM Clients

### OpenAI

```python
from ai_agent.clients import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000
)
```

### Anthropic (Claude)

```python
from ai_agent.clients import AnthropicClient

client = AnthropicClient(
    api_key="sk-ant-...",
    model="claude-3-opus-20240229"
)
```

### Local Models (Ollama)

```python
from ai_agent.clients import OllamaClient

client = OllamaClient(
    model="llama2",
    base_url="http://localhost:11434"
)
```

### Azure OpenAI

```python
from ai_agent.clients import AzureOpenAIClient

client = AzureOpenAIClient(
    api_key="your-key",
    api_version="2024-02-01",
    endpoint="https://your-resource.openai.azure.com/",
    deployment="gpt-4"
)
```

---

## 🧠 Memory Strategies

### ConversationBufferMemory

Stores all messages:

```python
from ai_agent.memory import ConversationBufferMemory

memory = ConversationBufferMemory(window_size=10)
```

### SummarizerMemory

Summarizes old messages:

```python
from ai_agent.memory import SummarizerMemory

memory = SummarizerMemory(
    client=client,
    summary_model="gpt-3.5-turbo",
    max_tokens=500
)
```

### TokenMemory

Limits by token count:

```python
from ai_agent.memory import TokenMemory

memory = TokenMemory(max_tokens=4000)
```

---

## 📡 Streaming

```python
# Stream responses token by token
for chunk in agent.run_stream("Explain quantum computing"):
    print(chunk, end="", flush=True)

# Or async
async for chunk in agent.run_async_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

---

## 🔄 Error Handling

```python
from ai_agent import Agent
from ai_agent.exceptions import (
    ToolNotFoundError,
    RateLimitError,
    AuthenticationError
)

try:
    response = agent.run("Complex query")
except RateLimitError:
    # Handle rate limiting
    print("Rate limited, waiting...")
except AuthenticationError:
    # Handle auth issues
    print("Check your API key")
```

---

## 📊 Logging & Observability

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

agent = Agent(client=client)

# View detailed logs
response = agent.run("Your query")

# Output includes:
# - Tool calls
# - Reasoning steps
# - Token usage
# - Latency metrics
```

---

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=ai_agent tests/

# Run specific test
pytest tests/test_agent.py::test_simple_run
```

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repo
2. Create a feature branch
3. Add tests for new features
4. Ensure tests pass
5. Submit a PR

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Resources

- [LangChain Documentation](https://python.langchain.com)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/gpt/function-calling)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [Anthropic Tool Use](https://docs.anthropic.com/claude/docs/tool-use)

---

## 🆘 Support

- Open an [Issue](https://github.com/Moses-main/ai-agent-toolkit/issues)
- Join our [Discord](https://discord.gg/your-invite)
- Check [Discussions](https://github.com/Moses-main/ai-agent-toolkit/discussions)

---

<p align="center">
  <strong>⭐ Star this repo if it's useful!</strong>
</p>
