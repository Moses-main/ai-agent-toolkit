# 📋 Agent Templates

Ready-to-use agent templates for different use cases.

## Available Templates

| Template | Use Case | Files |
|----------|----------|-------|
| **Simple** | Basic Q&A, no tools | `simple/agent.py` |
| **Tool Agent** | Tasks with external APIs | `tool_agent/agent.py` |
| **Conversational** | Chatbots with memory | `conversational/agent.py` |
| **ReAct** | Complex reasoning | `react/agent.py` |

## Quick Start

```bash
# Run a template
python templates/simple/agent.py

# Customize and run
cp templates/tool_agent/agent.py my_agent.py
# Edit my_agent.py with your API key
python my_agent.py
```

## Template Comparison

### Simple Agent
- No tools
- No memory (stateless)
- Best for: Simple queries

### Tool Agent  
- Tools for calculations, APIs, etc.
- Stateless
- Best for: Task-oriented agents

### Conversational Agent
- Memory of conversation
- Context-aware
- Best for: Chatbots, support

### ReAct Agent
- Step-by-step reasoning
- Iterative tool use
- Best for: Math, complex problems

## Customizing Templates

1. Copy a template
2. Add your API key
3. Modify system prompt
4. Add custom tools
5. Run!

```python
# Example: Adding a custom tool
@tool(name="my_tool", description="What it does")
def my_tool(param: str) -> str:
    # Your logic here
    return "result"
```
