"""Loop controllers for the Complete verb.

Controllers decide what action to take at each iteration of the agent loop.
They observe the current state and return an action (execute tools, instruct,
skip, complete, or fail).

Different controllers can be used for different models - weaker models may
need gentler, less frequent nudges while stronger models can be pushed harder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from llm_saia.core.classifier import LLMTaskStateClassifier, TaskState

if TYPE_CHECKING:
    from llm_saia.core.backend import AgentResponse, Message, ToolCall
    from llm_saia.core.config import Config, TerminalConfig


@dataclass
class Observation:
    """What the controller observes at each iteration."""

    response: AgentResponse
    messages: list[Message]
    iteration: int
    task: str
    tool_names: list[str]
    terminal_tool: str | None


class ActionKind(Enum):
    """Types of actions the controller can decide."""

    EXECUTE_TOOLS = "execute_tools"  # Normal flow - execute tool calls
    INSTRUCT = "instruct"  # Send a nudge/instruction message
    SKIP = "skip"  # Do nothing, let the loop continue
    COMPLETE = "complete"  # Task is done
    FAIL = "fail"  # Give up


@dataclass
class Action:
    """Action decided by the controller."""

    kind: ActionKind
    message: str | None = None  # For INSTRUCT
    output: str | None = None  # For COMPLETE/FAIL
    terminal_data: Any = None  # For COMPLETE via terminal tool
    terminal_tool: str | None = None  # Name of terminal tool if used
    tool_ids_to_execute: list[str] | None = None  # For EXECUTE_TOOLS - None means all
    reason: str = ""  # Why this action was chosen (for logging)


class LoopController(Protocol):
    """Protocol for loop controllers.

    Controllers observe the loop state and decide what action to take.
    This separates the decision logic from the loop mechanics.
    """

    async def decide(self, obs: Observation) -> Action:
        """Decide what action to take based on observation.

        Args:
            obs: Current observation (response, messages, iteration, etc.)

        Returns:
            Action to take (execute tools, instruct, skip, complete, fail)
        """
        ...

    def reset(self) -> None:
        """Reset controller state for a new task."""
        ...


@dataclass
class ControllerConfig:
    """Configuration for the default controller."""

    llm_config: Config  # Config for classifier LLM calls
    terminal: TerminalConfig | None = None  # Terminal tool configuration
    backoff_iterations: int = 5  # Iterations to wait after nudging
    max_confirmation_retries: int = 3  # Max retries for terminal tool confirmation


# Default nudge messages for each state
DEFAULT_NUDGES: dict[TaskState, str] = {
    TaskState.STUCK: (
        "You indicated you're stuck, but please continue working on the task. "
        "Try a different approach if needed. Use the available tools to proceed."
    ),
    TaskState.WANTS_CONTINUE: (
        "You indicated you want to continue but didn't use any tools. "
        "Please use the available tools to proceed with the task."
    ),
    TaskState.ASKING: "Yes, please proceed with the task using the available tools.",
}

DEFAULT_NUDGE_FALLBACK = "Please continue working on the task using the available tools."


@dataclass
class DefaultController:
    """Default controller with backoff and terminal tool support.

    This controller implements:
    - Terminal tool confirmation flow
    - State classification via LLM
    - Nudge backoff to avoid overwhelming the model
    - Contradiction detection in confirmations
    """

    config: ControllerConfig
    nudges: dict[TaskState, str] = field(default_factory=lambda: DEFAULT_NUDGES.copy())

    # Internal state
    _last_nudge_iteration: int = field(default=-100, init=False, repr=False)
    _pending_terminal: ToolCall | None = field(default=None, init=False, repr=False)
    _confirmation_retries: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        """Reset state for a new task."""
        self._last_nudge_iteration = -self.config.backoff_iterations
        self._pending_terminal = None
        self._confirmation_retries = 0

    async def decide(self, obs: Observation) -> Action:
        """Decide what action to take."""
        # 1. Check for terminal tool
        terminal_action = self._check_terminal_tool(obs)
        if terminal_action:
            return terminal_action

        # 2. Has tool calls → execute them
        if obs.response.tool_calls:
            return Action(ActionKind.EXECUTE_TOOLS, reason="has_tool_calls")

        # 3. No tool calls → classify and decide
        return await self._handle_no_tools(obs)

    def _check_terminal_tool(self, obs: Observation) -> Action | None:
        """Handle terminal tool flow (first call, confirmation, etc.)."""
        if not obs.terminal_tool:
            return None

        terminal_call = self._find_terminal_call(obs.response.tool_calls, obs.terminal_tool)

        # Check if this is a confirmation of pending terminal
        if self._pending_terminal and terminal_call:
            return self._handle_confirmation(terminal_call, obs)

        # First terminal call - request confirmation
        if terminal_call:
            return self._request_confirmation(terminal_call, obs)

        return None

    def _find_terminal_call(
        self, tool_calls: list[ToolCall] | None, terminal_tool: str
    ) -> ToolCall | None:
        """Find terminal tool call in the list."""
        if not tool_calls:
            return None
        for call in tool_calls:
            if call.name == terminal_tool:
                return call
        return None

    def _request_confirmation(self, call: ToolCall, obs: Observation) -> Action:
        """Request confirmation for terminal tool."""
        self._pending_terminal = call
        self._confirmation_retries = 0

        # Execute other tools if any (but not the terminal tool)
        other_calls = [c for c in (obs.response.tool_calls or []) if c.name != obs.terminal_tool]
        if other_calls:
            return Action(
                ActionKind.EXECUTE_TOOLS,
                tool_ids_to_execute=[c.id for c in other_calls],
                reason="terminal_with_other_tools",
            )

        # Only terminal call - send confirmation request
        msg = (
            f"You called `{obs.terminal_tool}` to indicate task completion. "
            f"Please call `{obs.terminal_tool}` again to confirm this is your final answer, "
            "or continue working if you have more to do."
        )
        return Action(ActionKind.INSTRUCT, message=msg, reason="terminal_confirmation_request")

    def _handle_confirmation(self, call: ToolCall, obs: Observation) -> Action:
        """Handle confirmation of terminal tool."""
        # Check for contradiction (says confirmed but has continuation signals)
        if self._has_contradiction(obs.response.content):
            self._confirmation_retries += 1
            if self._confirmation_retries >= self.config.max_confirmation_retries:
                self._pending_terminal = None
                return Action(
                    ActionKind.FAIL,
                    output=obs.response.content,
                    reason="confirmation_retries_exceeded",
                )

            msg = (
                "Your response contains contradictory signals - you confirmed completion "
                "but also indicated you want to continue. Please either continue working "
                "using the available tools, or call the terminal tool with a clear final answer."
            )
            return Action(ActionKind.INSTRUCT, message=msg, reason="contradiction_detected")

        # Confirmed - complete or fail based on status
        output = self._extract_terminal_output(call.arguments, obs.response.content)
        is_failure = self._is_terminal_failure(call.arguments)
        self._pending_terminal = None
        return Action(
            ActionKind.FAIL if is_failure else ActionKind.COMPLETE,
            output=output,
            terminal_data=call.arguments,
            terminal_tool=call.name,
            reason="terminal_confirmed_fail" if is_failure else "terminal_confirmed",
        )

    def _is_terminal_failure(self, arguments: dict[str, Any]) -> bool:
        """Check if terminal tool arguments indicate failure.

        Uses configured status_field if set, otherwise checks "status".
        Returns True if status value matches any failure value.
        """
        terminal = self.config.terminal
        # Get status field name (default: "status")
        status_field = (terminal.status_field if terminal else None) or "status"
        status_value = arguments.get(status_field, "")

        if not status_value:
            return False

        # Get failure values (default if not configured)
        failure_values = terminal.failure_values if terminal else ("stuck", "failed", "error")

        # Check against failure values (case-insensitive)
        status_lower = str(status_value).lower()
        return any(fv in status_lower for fv in failure_values)

    def _extract_terminal_output(self, arguments: dict[str, Any], fallback: str) -> str:
        """Extract output from terminal tool arguments.

        Uses the configured output_field if set.
        Falls back to checking common field names, then response content.
        """
        terminal = self.config.terminal
        # Use configured field if set
        field = terminal.output_field if terminal else None
        if field and field in arguments and arguments[field]:
            return str(arguments[field])

        # Fallback: check common field names
        common_fields = ("conclusion", "output", "result", "answer", "response")
        for field in common_fields:
            if field in arguments and arguments[field]:
                return str(arguments[field])

        return fallback

    def _has_contradiction(self, content: str) -> bool:
        """Check if content has continuation signals (contradiction)."""
        if not content:
            return False
        content_lower = content.lower()
        signals = (
            "let me ",
            "i will ",
            "i'll ",
            "let's ",
            "shall i ",
            "should i ",
            "would you like me to",
        )
        return any(s in content_lower for s in signals)

    async def _handle_no_tools(self, obs: Observation) -> Action:
        """Handle response with no tool calls."""
        classifier = LLMTaskStateClassifier(self.config.llm_config)
        result = await classifier.classify(obs.task, obs.response.content, obs.tool_names)

        if result.state == TaskState.COMPLETED:
            return Action(
                ActionKind.COMPLETE,
                output=obs.response.content,
                reason=f"classified_complete:{result.reason}",
            )

        # Check backoff
        iterations_since_nudge = obs.iteration - self._last_nudge_iteration
        if iterations_since_nudge < self.config.backoff_iterations:
            return Action(
                ActionKind.SKIP,
                reason=f"backoff:{iterations_since_nudge}/{self.config.backoff_iterations}",
            )

        # Send nudge
        self._last_nudge_iteration = obs.iteration
        nudge = self.nudges.get(result.state, DEFAULT_NUDGE_FALLBACK)
        return Action(
            ActionKind.INSTRUCT,
            message=nudge,
            reason=f"nudge:{result.state.value}",
        )
