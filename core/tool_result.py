"""Typed result passed from MCP clients into model tool loops."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """A tool result plus control metadata that must not be shown to the model."""

    text: str
    is_error: bool = False
    silent_end_turn: bool = False
