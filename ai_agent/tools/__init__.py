"""Tool System for AI Agents."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, get_type_hints
import inspect
import json


class BaseTool(ABC):
    """Base class for tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

    def get_schema(self) -> dict:
        """Get the tool schema for LLM function calling."""
        hints = get_type_hints(self.execute)
        
        # Get function signature
        sig = inspect.signature(self.execute)
        params = sig.parameters
        
        properties = {}
        required = []
        
        for param_name, param in params.items():
            if param_name == 'self':
                continue
                
            param_type = hints.get(param_name, str)
            type_str = self._python_type_to_json(param_type)
            
            properties[param_name] = {
                "type": type_str,
                "description": f"Parameter {param_name}"
            }
            
            if param.default is inspect.Parameter.empty:
                required.append(param_name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def _python_type_to_json(self, python_type) -> str:
        """Convert Python type to JSON schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(python_type, "string")


class Tool(BaseTool):
    """Tool wrapper for functions."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict] = None,
    ):
        self._name = name
        self._description = description
        self._func = func
        self._parameters = parameters or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def execute(self, **kwargs) -> Any:
        """Execute the function."""
        return self._func(**kwargs)

    def get_schema(self) -> dict:
        """Get tool schema."""
        if self._parameters:
            return {
                "type": "function",
                "function": {
                    "name": self._name,
                    "description": self._description,
                    "parameters": self._parameters
                }
            }
        return super().get_schema()


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]

    def clear(self) -> None:
        """Clear all tools."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)


# Global registry
tool_registry = ToolRegistry()


# Decorator for creating tools
def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict] = None,
) -> Callable:
    """
    Decorator to create a tool from a function.
    
    Example:
        @tool(name="multiply", description="Multiply two numbers")
        def multiply(a: float, b: float) -> float:
            return a * b
    """
    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Function {func.__name__}"
        
        return Tool(
            name=tool_name,
            description=tool_desc,
            func=func,
            parameters=parameters,
        )
    
    return decorator


# Built-in tools

@tool(
    name="calculator",
    description="Perform basic mathematical calculations",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5', '2 ** 3')"
            }
        },
        "required": ["expression"]
    }
)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        # Safe evaluation - only allow basic math
        allowed_names = {
            "abs": abs,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
        }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="get_current_date",
    description="Get the current date and time",
    parameters={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Format string (default: '%Y-%m-%d %H:%M:%S')"
            }
        }
    }
)
def get_current_date(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Get current date/time."""
    from datetime import datetime
    return datetime.now().strftime(format)


@tool(
    name="get_weather",
    description="Get weather information for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or location"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit (default: celsius)"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather for a location (placeholder - add API key for real data)."""
    # This is a placeholder - in production, you'd use a weather API
    return f"Weather in {location}: Sunny, 22°C (placeholder - add API for real data)"


@tool(
    name="search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
)
def search(query: str) -> str:
    """Search the web (placeholder - add API for real search)."""
    # This is a placeholder - in production, you'd use a search API
    return f"Search results for '{query}': (placeholder - add search API)"


@tool(
    name="text_length",
    description="Get the length of a text string",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to measure"
            }
        },
        "required": ["text"]
    }
)
def text_length(text: str) -> dict:
    """Get text length and character count."""
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": len(text.splitlines()),
    }


# Export everything
__all__ = [
    "BaseTool",
    "Tool",
    "ToolRegistry",
    "tool_registry",
    "tool",
    "calculator",
    "get_current_date",
    "get_weather",
    "search",
    "text_length",
]
