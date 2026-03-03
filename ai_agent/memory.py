"""Memory Management for AI Agents."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json


class BaseMemory(ABC):
    """Base class for memory implementations."""

    @abstractmethod
    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages."""
        pass


class ConversationBufferMemory(BaseMemory):
    """
    Simple memory that stores all messages.
    
    Example:
        memory = ConversationBufferMemory()
        memory.add_message({"role": "user", "content": "Hello"})
        messages = memory.get_messages()
    """

    def __init__(self, window_size: Optional[int] = None):
        """
        Initialize memory.
        
        Args:
            window_size: If set, only keep last N messages
        """
        self.messages: List[Dict[str, Any]] = []
        self.window_size = window_size

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message to memory."""
        self.messages.append(message)

        # Apply window size limit
        if self.window_size and len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []

    def __len__(self) -> int:
        return len(self.messages)


class SummarizerMemory(BaseMemory):
    """
    Memory that summarizes old messages to save tokens.
    
    Example:
        memory = SummarizerMemory(client=client)
        # After many messages, old ones get summarized
    """

    def __init__(
        self,
        client: Any = None,
        summary_model: str = "gpt-3.5-turbo",
        max_messages_before_summary: int = 10,
        summary_prompt: str = "Summarize this conversation concisely:",
    ):
        """
        Initialize summarizer memory.
        
        Args:
            client: LLM client for creating summaries
            summary_model: Model to use for summarization
            max_messages: Number of messages before summarizing
            summary_prompt: Prompt to use for summarization
        """
        self.client = client
        self.summary_model = summary_model
        self.max_messages = max_messages_before_summary
        self.summary_prompt = summary_prompt

        self.messages: List[Dict[str, Any]] = []
        self.summary: Optional[str] = None

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message and potentially summarize."""
        self.messages.append(message)

        # Summarize if we have too many messages
        if len(self.messages) > self.max_messages:
            self._summarize()

    def _summarize(self) -> None:
        """Summarize old messages."""
        if not self.client:
            return

        # Get messages to summarize (keep recent ones)
        to_summarize = self.messages[:-5]  # Keep last 5

        if not to_summarize:
            return

        # Create summary prompt
        conversation_text = "\n".join(
            f"{msg['role']}: {msg['content'][:200]}..."
            for msg in to_summarize
        )

        summary_prompt = f"{self.summary_prompt}\n\n{conversation_text}"

        try:
            response = self.client.chat(
                messages=[{"role": "user", "content": summary_prompt}],
                model=self.summary_model,
                max_tokens=500,
            )

            self.summary = response.get("content", "")
        except Exception:
            # If summarization fails, just trim messages
            self.messages = self.messages[-self.max_messages:]
            return

        # Keep only recent messages + summary
        self.messages = [
            {"role": "system", "content": f"Previous conversation summary: {self.summary}"}
        ] + self.messages[-5:]

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages including summary."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages and summary."""
        self.messages = []
        self.summary = None


class TokenMemory(BaseMemory):
    """
    Memory that limits messages by token count.
    
    Example:
        memory = TokenMemory(max_tokens=4000)
        # Automatically removes oldest messages when limit exceeded
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        tokens_per_message: int = 4,  # Approximate overhead
    ):
        """
        Initialize token memory.
        
        Args:
            max_tokens: Maximum tokens to store
            tokens_per_message: Overhead per message
        """
        self.max_tokens = max_tokens
        self.tokens_per_message = tokens_per_message
        self.messages: List[Dict[str, Any]] = []

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simple approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def add_message(self, message: Dict[str, Any]) -> None:
        """Add a message, removing old ones if needed."""
        message_text = message.get("content", "")
        message_tokens = self._count_tokens(message_text) + self.tokens_per_message

        self.messages.append(message)

        # Remove old messages if over limit
        while self._get_total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def _get_total_tokens(self) -> int:
        """Get total estimated tokens."""
        total = 0
        for msg in self.messages:
            total += self._count_tokens(msg.get("content", "")) + self.tokens_per_message
        return total

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []


# Export all memory types
__all__ = [
    "BaseMemory",
    "ConversationBufferMemory",
    "SummarizerMemory",
    "TokenMemory",
]
