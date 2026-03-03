"""
Conversational Agent Template

Best for: Chatbots, customer support, context-aware conversations

Usage:
    python templates/conversational/agent.py
"""

from ai_agent import Agent
from ai_agent.clients import OpenAIClient
from ai_agent.memory import ConversationBufferMemory


# Initialize client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Create memory (stores conversation history)
# Options:
# - ConversationBufferMemory(window_size=10)  # Keep last 10 messages
# - TokenMemory(max_tokens=4000)             # Limit by tokens
# - SummarizerMemory(client=client)           # Summarize old messages
memory = ConversationBufferMemory(window_size=20)

# Create conversational agent
agent = Agent(
    client=client,
    model="gpt-4",
    memory=memory,
    system_prompt="""You are a friendly customer support assistant.

Your traits:
- Patient and understanding
- Clear and concise
- Empathetic
- Professional

You remember context from the conversation.""",
    temperature=0.7,
)


# ============================================
# Run conversation (history is maintained)
# ============================================

if __name__ == "__main__":
    print("🤖 Conversational Agent")
    print("=" * 50)
    print("I remember our conversation!")
    print("Type 'clear' to reset memory")
    print("Type 'history' to see conversation")
    print("Type 'exit' to quit")
    print()

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if user_input.lower() == "clear":
            agent.reset_memory()
            print("Memory cleared!")
            continue
        
        if user_input.lower() == "history":
            print("\n--- Conversation History ---")
            for msg in agent.chat_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]
                print(f"{role}: {content}...")
            print("--- End History ---\n")
            continue
        
        response = agent.run(user_input)
        print(f"\nAssistant: {response}")
