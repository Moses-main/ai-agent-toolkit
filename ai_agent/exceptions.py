"""Custom exceptions for AI Agent Toolkit."""


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class ToolNotFoundError(AgentError):
    """Raised when a requested tool is not found."""
    pass


class RateLimitError(AgentError):
    """Raised when rate limit is exceeded."""
    pass


class AuthenticationError(AgentError):
    """Raised when authentication fails."""
    pass


class InvalidResponseError(AgentError):
    """Raised when the LLM returns an invalid response."""
    pass


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails."""
    pass
