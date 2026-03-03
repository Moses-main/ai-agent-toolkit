"""LLM Client Interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional
import os


class BaseClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> dict:
        """Send a chat request."""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Send a streaming chat request."""
        pass


class OpenAIClient(BaseClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        **kwargs,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            **kwargs,
        )
        self.default_model = "gpt-4"

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> dict:
        """Send a chat request to OpenAI."""
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if tools:
            params["tools"] = tools

        if stream:
            # Return the stream iterator
            return self.client.chat.completions.create(**params)

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response)

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Send a streaming chat request."""
        params = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            params["max_tokens"] = max_tokens

        if tools:
            params["tools"] = tools

        response = self.client.chat.completions.create(**params)

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _parse_response(self, response) -> dict:
        """Parse OpenAI response to standard format."""
        choice = response.choices[0]
        message = choice.message

        result = {
            "content": message.content or "",
            "role": message.role,
            "finish_reason": choice.finish_reason,
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return result


class AnthropicClient(BaseClient):
    """Anthropic API client (Claude)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs,
        )
        self.default_model = "claude-3-opus-20240229"

    def chat(
        self,
        messages: list[dict],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> dict:
        """Send a chat request to Anthropic."""
        # Convert messages format
        system = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        params = {
            "model": model or self.default_model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }

        if system:
            params["system"] = system

        if tools:
            # Anthropic uses tools differently
            params["tools"] = tools

        if stream:
            return self.client.messages.stream(**params)

        response = self.client.messages.create(**params)

        return {
            "content": response.content[0].text if response.content else "",
            "role": "assistant",
            "finish_reason": "stop",
        }

    async def chat_stream(
        self,
        messages: list[dict],
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Send a streaming chat request."""
        # Convert messages format
        system = None
        filtered_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        params = {
            "model": model or self.default_model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
            "stream": True,
        }

        if system:
            params["system"] = system

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text


class OllamaClient(BaseClient):
    """Ollama local model client."""

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        self.model = model
        self.base_url = base_url
        self.default_params = kwargs

    def chat(
        self,
        messages: list[dict],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        stream: bool = False,
    ) -> dict:
        """Send a chat request to Ollama."""
        import requests

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        if tools:
            payload["tools"] = tools

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()

        if stream:
            return data  # Return the iterator

        return {
            "content": data.get("message", {}).get("content", ""),
            "role": data.get("message", {}).get("role", "assistant"),
            "finish_reason": "stop" if not data.get("done", False) else "complete",
        }

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "llama2",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[dict]] = None,
    ) -> AsyncIterator[str]:
        """Send a streaming chat request to Ollama."""
        import requests

        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens:
            payload["options"] = {"num_predict": max_tokens}

        if tools:
            payload["tools"] = tools

        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8")
                import json

                chunk = json.loads(data)
                if "message" in chunk and "content" in chunk["message"]:
                    content = chunk["message"]["content"]
                    if content:
                        yield content


# Convenience function
def get_client(client_type: str = "openai", **kwargs) -> BaseClient:
    """Get a client by type."""
    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "ollama": OllamaClient,
    }

    if client_type not in clients:
        raise ValueError(f"Unknown client type: {client_type}")

    return clients[client_type](**kwargs)
