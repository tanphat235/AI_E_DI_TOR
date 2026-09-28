from __future__ import annotations


class PublishError(Exception):
    """A failure the user can act on. ``code`` is stable; ``hint`` says what to do."""

    def __init__(self, code: str, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint

    def __str__(self) -> str:
        base = f"{self.code}: {self.args[0]}"
        return f"{base} ({self.hint})" if self.hint else base
