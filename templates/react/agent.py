"""
ReAct Agent Template

Best for: Complex reasoning, multi-step math problems, chain-of-thought tasks

The ReAct (Reasoning + Acting) agent:
1. Thinks about the problem
2. Takes an action (uses a tool)
3. Observes the result
4. Repeats until it has an answer

Usage:
    python templates/react/agent.py
"""

from ai_agent import ReActAgent
from ai_agent.clients import OpenAIClient
from ai_agent.tools import tool


# ============================================
# Define tools for reasoning
# ============================================

@tool(name="calculate", description="Calculate a mathematical expression")
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    allowed = {"abs": abs, "max": max, "min": min, "pow": pow, "round": round}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool(name="search", description="Search for information online")
def search(query: str) -> str:
    """Search the web (placeholder)."""
    return f"Search results for '{query}': [Demo - add search API]" 


# ============================================
# Agent setup
# ============================================

# Initialize client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4"  # Use GPT-4 for best reasoning
)

# Create ReAct agent
agent = ReActAgent(
    client=client,
    model="gpt-4",
    tools=[calculate, search],
    max_iterations=10,
    system_prompt="""You are an AI assistant that thinks step by step.

For each question:
1. Think about what you need to know
2. Use tools to get information
3. Reason about the results
4. Provide your final answer

Use this format:
Thought: [your reasoning]
Action: tool_name with arg=value
Observation: [result]

When you know the answer:
Thought: I now know the answer
Action: Finish with [your final answer]"""
)


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    print("🧠 ReAct Agent (Reasoning + Acting)")
    print("=" * 50)
    print("The agent will think, act, and observe step by step.")
    print()

    questions = [
        "What is 15 + 27?",
        "What is 5 factorial (5!)?",
        "If I have 3 items costing $10 each plus $5 tax, what's the total?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)
        
        result = agent.run(question)
        
        print(f"\n✅ Answer: {result.final_answer}")
        print(f"📊 Iterations: {result.iterations}")
        print("\n" + "=" * 50)

    # Interactive mode
    print("\nInteractive mode (type 'exit' to quit):")
    
    while True:
        question = input("\n❓ Question: ")
        
        if question.lower() == "exit":
            break
        
        result = agent.run(question)
        print(f"\n✅ Answer: {result.final_answer}")
