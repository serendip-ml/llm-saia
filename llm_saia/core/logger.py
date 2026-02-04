"""Logger interface for SAIA instrumentation."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SAIALogger(Protocol):
    """Logger interface for SAIA instrumentation.

    Implementations (like appinfra's Logger) satisfy this via structural typing.
    No explicit inheritance required - just implement the methods.
    """

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at DEBUG level."""
        ...

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at INFO level."""
        ...

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """Log at WARNING level."""
        ...


class NullLogger:
    """No-op logger for when logging is disabled.

    Satisfies SAIALogger protocol without doing anything.
    """

    def debug(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def info(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass

    def warning(self, msg: str, *, extra: dict[str, Any] | None = None) -> None:
        """No-op."""
        pass
