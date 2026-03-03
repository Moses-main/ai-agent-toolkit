"""
Tool-using Agent Template

Best for: Tasks requiring external actions (APIs, calculations, etc.)

Usage:
    python templates/tool_agent/agent.py
"""

from ai_agent import Agent
from ai_agent.clients import OpenAIClient
from ai_agent.tools import tool


# ============================================
# Define your custom tools here
# ============================================

@tool(
    name="calculator",
    description="Perform mathematical calculations"
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = {"abs": abs, "max": max, "min": min, "pow": pow, "round": round}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(
    name="get_weather",
    description="Get weather information for a location"
)
def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get weather for a location.
    
    Args:
        location: City name
        unit: celsius or fahrenheit
    """
    # Replace with actual weather API
    return f"Weather in {location}: 22°C, Sunny (demo)"


@tool(
    name="search_wiki",
    description="Search Wikipedia for information"
)
def search_wiki(query: str) -> str:
    """Search Wikipedia (placeholder - add real API)."""
    # Use wikipedia-api package in production
    return f"Results for '{query}': [Demo result]"


# ============================================
# Agent setup
# ============================================

# Initialize client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Register tools
TOOLS = [calculator, get_weather, search_wiki]

# Create agent with tools
agent = Agent(
    client=client,
    model="gpt-4",
    tools=TOOLS,
    system_prompt="""You are a helpful assistant with access to tools.

Available tools:
- calculator: Perform calculations
- get_weather: Get weather info
- search_wiki: Search Wikipedia

Use tools when needed to answer questions accurately."""
)


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    print("🤖 Tool-using Agent")
    print("=" * 50)
    print("Available tools: calculator, get_weather, search_wiki")
    print()

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        response = agent.run(user_input)
        print(f"\nAssistant: {response}")
