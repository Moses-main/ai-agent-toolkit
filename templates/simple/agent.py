"""
Simple Chat Agent Template

Best for: Basic Q&A, simple conversations, no tools needed.

Usage:
    python templates/simple/agent.py
"""

from ai_agent import Agent
from ai_agent.clients import OpenAIClient

# Initialize client (change provider as needed)
client = OpenAIClient(
    api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
    model="gpt-4"
)

# Create simple chat agent
agent = Agent(
    client=client,
    model="gpt-4",
    system_prompt="""You are a helpful AI assistant.
    
Guidelines:
- Be concise and friendly
- Provide accurate information
- If unsure, say so""",
    temperature=0.7,
)

# Run conversation
print("🤖 Simple Chat Agent")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    
    response = agent.run(user_input)
    print(f"\nAssistant: {response}")
