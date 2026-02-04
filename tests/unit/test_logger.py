"""Tests for logger protocol and integration."""

from typing import Any

import pytest

from llm_saia import SAIA, NullLogger, SAIALogger
from llm_saia.core.types import AgentResponse, ToolCall, ToolDef
from tests.unit.conftest import MockBackend

pytestmark = pytest.mark.unit


class RecordingLogger:
    """Test logger that records all calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("debug", msg, extra))

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("info", msg, extra))

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        self.calls.append(("warning", msg, extra))


class TestSAIALogger:
    def test_protocol_satisfaction(self) -> None:
        """RecordingLogger satisfies SAIALogger protocol."""
        logger = RecordingLogger()
        assert isinstance(logger, SAIALogger)

    def test_null_logger_satisfies_protocol(self) -> None:
        """NullLogger satisfies SAIALogger protocol."""
        logger = NullLogger()
        assert isinstance(logger, SAIALogger)

    def test_null_logger_no_op(self) -> None:
        """NullLogger does nothing."""
        logger = NullLogger()
        logger.debug("test")
        logger.info("test", extra={"key": "value"})
        logger.warning("test")
        # No exception raised, no output


class TestLoggerIntegration:
    def test_saia_accepts_logger(self, mock_backend: MockBackend) -> None:
        """SAIA accepts a logger parameter."""
        logger = RecordingLogger()
        saia = SAIA(backend=mock_backend, lg=logger)
        assert saia._lg is logger

    def test_saia_without_logger(self, mock_backend: MockBackend) -> None:
        """SAIA works without a logger."""
        saia = SAIA(backend=mock_backend)
        assert saia._lg is None

    def test_logger_preserved_in_with_methods(self, mock_backend: MockBackend) -> None:
        """Logger is preserved through with_* methods."""
        logger = RecordingLogger()
        saia = SAIA(backend=mock_backend, lg=logger)

        saia2 = saia.with_max_iterations(5)
        assert saia2._lg is logger

        saia3 = saia.with_single_call()
        assert saia3._lg is logger

    async def test_logger_called_during_verb_with_tools(self, mock_backend: MockBackend) -> None:
        """Logger is called during verb execution when tools are configured."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        await saia.ask("artifact", "question")

        # Should have log calls from the loop
        assert len(logger.calls) > 0
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages

    async def test_loop_logs_start_and_response(self, mock_backend: MockBackend) -> None:
        """Loop logs start and LLM response."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue a tool call followed by completion
        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="done", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("do something")

        # Check for expected log messages
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        assert "executing tool" in messages

    async def test_loop_logs_limit_reached(self, mock_backend: MockBackend) -> None:
        """Loop logs when iteration limit is reached."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue multiple tool calls that exceed max_iterations
        for _ in range(5):
            mock_backend.queue_response(
                AgentResponse(
                    content="calling tool",
                    tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                    stop_reason="tool_use",
                )
            )

        await saia.with_max_iterations(2).instruct("do something")

        # Check for limit warning
        messages = [call[1] for call in logger.calls]
        assert "verb loop limit reached" in messages

        # Check limit type is correct
        limit_call = [c for c in logger.calls if c[1] == "verb loop limit reached"][0]
        assert limit_call[2] is not None
        assert limit_call[2]["limit_type"] == "max_iterations"

    async def test_tool_error_logged(self, mock_backend: MockBackend) -> None:
        """Tool execution errors are logged."""
        logger = RecordingLogger()
        tools = [ToolDef(name="failing_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            raise ValueError("Tool failed!")

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="failing_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="done", tool_calls=[], stop_reason="end_turn")
        )

        await saia.instruct("do something")

        # Check for error warning
        warning_calls = [c for c in logger.calls if c[0] == "warning"]
        assert len(warning_calls) > 0
        error_call = [c for c in warning_calls if c[1] == "tool execution failed"][0]
        assert error_call[2] is not None
        assert error_call[2]["tool_name"] == "failing_tool"
        assert isinstance(error_call[2]["exception"], ValueError)

    async def test_complete_verb_logs(self, mock_backend: MockBackend) -> None:
        """Complete verb logs start, responses, and limit reached."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue tool calls that exceed max_iterations (Complete has its own loop)
        for _ in range(5):
            mock_backend.queue_response(
                AgentResponse(
                    content="calling tool",
                    tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                    stop_reason="tool_use",
                )
            )

        await saia.with_max_iterations(2).complete("do a task")

        # Verify Complete logged properly
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        assert "verb loop limit reached" in messages

        # Check limit type
        limit_call = [c for c in logger.calls if c[1] == "verb loop limit reached"][0]
        assert limit_call[2] is not None
        assert limit_call[2]["limit_type"] == "max_iterations"

    async def test_complete_verb_logs_successful_completion(
        self, mock_backend: MockBackend
    ) -> None:
        """Complete verb logs when task completes successfully."""
        logger = RecordingLogger()
        tools = [ToolDef(name="test_tool", description="test", parameters={})]

        async def executor(name: str, args: dict[str, Any]) -> str:
            return "result"

        saia = SAIA(backend=mock_backend, tools=tools, executor=executor, lg=logger)

        # Queue: tool call -> completion response -> confirm says done
        mock_backend.queue_response(
            AgentResponse(
                content="calling tool",
                tool_calls=[ToolCall(id="1", name="test_tool", arguments={})],
                stop_reason="tool_use",
            )
        )
        mock_backend.queue_response(
            AgentResponse(content="task done", tool_calls=[], stop_reason="end_turn")
        )
        # ConfirmResult default is confirmed=True

        result = await saia.complete("do a task")

        assert result.completed is True
        messages = [call[1] for call in logger.calls]
        assert "verb loop started" in messages
        assert "llm response received" in messages
        # Should NOT have limit reached since it completed successfully
        assert "verb loop limit reached" not in messages
