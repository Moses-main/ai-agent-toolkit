"""Example: Simple Agent."""

from ai_agent import Agent, OpenAIClient
from ai_agent.tools import calculator, get_current_date, text_length

# Initialize client
client = OpenAIClient(
    api_key="your-openai-api-key",  # Or set OPENAI_API_KEY env var
    model="gpt-4"
)

# Create agent
agent = Agent(
    client=client,
    model="gpt-4",
    system_prompt="You are a helpful assistant.",
    tools=[calculator, get_current_date, text_length]
)

# Run queries
print("=" * 50)
print("Example 1: Calculator tool")
print("=" * 50)
response = agent.run("What is 123 * 456?")
print(f"Response: {response}")

print("\n" + "=" * 50)
print("Example 2: Date tool")
print("=" * 50)
response = agent.run("What is today's date?")
print(f"Response: {response}")

print("\n" + "=" * 50)
print("Example 3: Text length tool")
print("=" * 50)
response = agent.run("How many words are in 'Hello world, this is a test'?")
print(f"Response: {response}")

print("\n" + "=" * 50)
print("Example 4: Multi-tool query")
print("=" * 50)
response = agent.run("What's 50 + 50, and what's today's date?")
print(f"Response: {response}")
