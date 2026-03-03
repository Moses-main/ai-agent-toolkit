"""Core Agent Implementation."""

from typing import Any, AsyncIterator, Callable, Optional
import logging
import json
from .client import BaseClient
from .memory import BaseMemory, ConversationBufferMemory
from .tools import Tool, tool_registry
from .exceptions import AgentError, ToolNotFoundError

logger = logging.getLogger(__name__)


class Agent:
    """
    AI Agent with tool calling and memory support.
    
    Example:
        agent = Agent(
            client=OpenAIClient(api_key="sk-..."),
            model="gpt-4",
            tools=[calculator, search]
        )
        response = agent.run("What's 5 + 3?")
    """

    def __init__(
        self,
        client: BaseClient,
        model: str = "gpt-4",
        tools: Optional[list[Callable]] = None,
        memory: Optional[BaseMemory] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.memory = memory or ConversationBufferMemory()
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Register tools
        self._tool_registry = tool_registry
        for tool in self.tools:
            self._tool_registry.register(tool)

    def _build_messages(self, user_input: str) -> list[dict]:
        """Build message list with memory."""
        messages = []

        # Add system prompt
        messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history
        history = self.memory.get_messages()
        messages.extend(history)

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        return messages

    def _get_tool_schemas(self) -> list[dict]:
        """Get tool schemas for the LLM."""
        schemas = []
        for tool in self._tool_registry.get_all():
            schemas.append(tool.get_schema())
        return schemas

    def _execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """Execute a tool by name."""
        tool = self._tool_registry.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool '{tool_name}' not found")

        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        result = tool.execute(**arguments)
        logger.info(f"Tool result: {result}")

        return result

    def _should_use_tools(self) -> bool:
        """Check if tools are available."""
        return len(self._tool_registry.get_all()) > 0

    def run(self, user_input: str) -> str:
        """
        Run the agent with a user input.
        
        Args:
            user_input: The user's input/query
            
        Returns:
            The agent's response as a string
        """
        messages = self._build_messages(user_input)
        tool_schemas = self._get_tool_schemas() if self._should_use_tools() else None

        # First LLM call
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            tools=tool_schemas,
        )

        # Handle tool calls
        while response.get("tool_calls"):
            # Add assistant message with tool calls
            tool_call_message = {
                "role": "assistant",
                "content": response.get("content", ""),
                "tool_calls": response["tool_calls"],
            }
            self.memory.add_message(tool_call_message)
            messages.append(tool_call_message)

            # Execute each tool call
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                arguments = json.loads(tool_call["function"]["arguments"])

                try:
                    result = self._execute_tool(tool_name, arguments)
                except Exception as e:
                    result = f"Error: {str(e)}"

                # Add tool result to messages
                tool_result_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                }
                self.memory.add_message(tool_result_message)
                messages.append(tool_result_message)

            # Make another LLM call with tool results
            response = self.client.chat(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=tool_schemas,
            )

        # Get final response
        final_content = response.get("content", "")

        # Add assistant's final response to memory
        self.memory.add_message({
            "role": "assistant",
            "content": final_content
        })

        return final_content

    def run_stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Run the agent with streaming responses.
        
        Note: Tool calling with streaming is more complex and
        currently returns the full response after tool execution.
        """
        # For streaming, we currently fall back to non-streaming
        # for tool-enabled agents
        if self._should_use_tools():
            result = self.run(user_input)
            yield result
            return

        messages = self._build_messages(user_input)

        for token in self.client.chat_stream(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ):
            yield token

    async def run_async(self, user_input: str) -> str:
        """Async version of run."""
        return self.run(user_input)

    def reset_memory(self):
        """Clear the agent's memory."""
        self.memory.clear()

    @property
    def chat_history(self) -> list[dict]:
        """Get chat history."""
        return self.memory.get_messages()


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent.
    
    This agent iteratively:
    1. Thinks about the current state
    2. Takes an action (uses a tool)
    3. Observes the result
    4. Repeats until it has an answer
    
    Example:
        agent = ReActAgent(client=client, max_iterations=10)
        result = agent.run("What's 5! factorial?")
    """

    def __init__(
        self,
        client: BaseClient,
        model: str = "gpt-4",
        tools: Optional[list[Callable]] = None,
        max_iterations: int = 10,
        system_prompt: str = "You are a helpful assistant that thinks step by step.",
    ):
        self.client = client
        self.model = model
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt

        # Register tools
        self._tool_registry = tool_registry
        for tool in self.tools:
            self._tool_registry.register(tool)

    def _get_next_action_prompt(self, thought: str, action: str, observation: str) -> str:
        """Generate prompt for next step."""
        return f"""
Previous steps:
Thought: {thought}
Action: {action}
Observation: {observation}

What's your next thought? If you have the answer, provide it. Otherwise, decide on the next action.
"""

    def run(self, question: str) -> "ReActResult":
        """
        Run the ReAct agent.
        
        Args:
            question: The question to answer
            
        Returns:
            ReActResult with final answer and trace
        """
        trace = []

        # Initial prompt
        current_prompt = f"""
Question: {question}

Think step by step. Use the format:
Thought: [your reasoning]
Action: [tool_name] with [arguments]
Observation: [result of action]

When you know the answer:
Thought: I now know the answer
Action: Finish with [your final answer]
"""

        for i in range(self.max_iterations):
            logger.info(f"Iteration {i + 1}/{self.max_iterations}")

            # Get LLM response
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": current_prompt}
                ],
                model=self.model,
            )

            content = response.get("content", "")
            trace.append(content)

            # Parse the response
            lines = content.strip().split("\n")
            thought = ""
            action = ""

            for line in lines:
                if line.startswith("Thought:"):
                    thought = line[8:].strip()
                elif line.startswith("Action:"):
                    action = line[7:].strip()

            # Check if finished
            if "Finish with" in action or "final answer" in thought.lower():
                # Extract final answer
                for line in lines:
                    if line.startswith("Action:") and "Finish with" in line:
                        final_answer = line.split("Finish with")[-1].strip()
                        return ReActResult(
                            answer=final_answer,
                            trace=trace,
                            iterations=i + 1
                        )

            # Execute action if there is one
            observation = ""
            if action and action != "Finish with":
                try:
                    # Parse action (format: "tool_name with arg1=value1, arg2=value2")
                    if " with " in action:
                        tool_name, args_str = action.split(" with ", 1)
                        # Parse arguments
                        arguments = {}
                        if args_str:
                            for arg in args_str.split(", "):
                                if "=" in arg:
                                    key, value = arg.split("=", 1)
                                    arguments[key.strip()] = value.strip()

                        # Execute tool
                        tool = self._tool_registry.get(tool_name.strip())
                        if tool:
                            result = tool.execute(**arguments)
                            observation = str(result)
                        else:
                            observation = f"Tool '{tool_name}' not found"
                    else:
                        observation = "Invalid action format"
                except Exception as e:
                    observation = f"Error: {str(e)}"

            # Update prompt for next iteration
            current_prompt = self._get_next_action_prompt(thought, action, observation)

        # Max iterations reached
        return ReActResult(
            answer="Maximum iterations reached without finding an answer",
            trace=trace,
            iterations=self.max_iterations
        )


class ReActResult:
    """Result from a ReAct agent run."""

    def __init__(self, answer: str, trace: list[str], iterations: int):
        self.answer = answer
        self.trace = trace
        self.iterations = iterations

    def __str__(self) -> str:
        return self.answer

    @property
    def final_answer(self) -> str:
        """Get the final answer."""
        return self.answer
