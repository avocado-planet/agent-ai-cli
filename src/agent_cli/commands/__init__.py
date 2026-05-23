"""Slash command system - registry and built-in commands."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_cli.repl import AgentREPL


class SlashCommand(ABC):
    """Base class for slash commands."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def execute(self, args: str, repl: AgentREPL) -> str | None:
        """Execute the command. Return a message to display, or None."""
        ...


class CommandRegistry:
    """Manages slash command registration and dispatch."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd

    @property
    def command_names(self) -> list[str]:
        return sorted(self._commands.keys())

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)

    def parse_and_execute(self, user_input: str, repl: AgentREPL) -> tuple[bool, str | None]:
        """
        Try to parse input as a slash command.
        Returns (was_command, result_message).
        """
        if not user_input.startswith("/"):
            return False, None

        parts = user_input[1:].split(maxsplit=1)
        cmd_name = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.get(cmd_name)
        if cmd is None:
            return True, f"Unknown command: /{cmd_name}. Type /help for available commands."

        result = cmd.execute(args, repl)
        return True, result


# --- Built-in Commands ---


class HelpCommand(SlashCommand):
    name = "help"
    description = "Show available commands"

    def execute(self, args: str, repl: AgentREPL) -> str:
        lines = ["Available commands:\n"]
        for cmd_name in repl.registry.command_names:
            cmd = repl.registry.get(cmd_name)
            lines.append(f"  /{cmd_name:<12} {cmd.description}")
        return "\n".join(lines)


class ClearCommand(SlashCommand):
    name = "clear"
    description = "Clear conversation history"

    def execute(self, args: str, repl: AgentREPL) -> str:
        repl.state.clear()
        return "Conversation history cleared."


class ModelCommand(SlashCommand):
    name = "model"
    description = "Show or switch model (e.g. /model gpt-4o)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        if not args.strip():
            return f"Current model: {repl.config.provider}/{repl.config.model}"
        repl.config.model = args.strip()
        repl.rebuild_llm()
        return f"Model switched to: {repl.config.model}"


class ProviderCommand(SlashCommand):
    name = "provider"
    description = "Switch provider (openai / anthropic)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        provider = args.strip().lower()
        if provider not in ("openai", "anthropic"):
            return f"Current: {repl.config.provider}. Usage: /provider openai|anthropic"
        repl.config.provider = provider
        repl.config.model = repl.config.MODEL_DEFAULTS[provider]
        repl.rebuild_llm()
        return f"Switched to {provider} (model: {repl.config.model})"


class SystemCommand(SlashCommand):
    name = "system"
    description = "Show or set system prompt"

    def execute(self, args: str, repl: AgentREPL) -> str:
        if not args.strip():
            return f"System prompt:\n{repl.config.system_prompt}"
        repl.config.system_prompt = args.strip()
        return "System prompt updated."


class TemperatureCommand(SlashCommand):
    name = "temperature"
    description = "Set temperature (0.0-2.0)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        if not args.strip():
            return f"Current temperature: {repl.config.temperature}"
        try:
            t = float(args.strip())
            if not 0.0 <= t <= 2.0:
                return "Temperature must be between 0.0 and 2.0"
            repl.config.temperature = t
            repl.rebuild_llm()
            return f"Temperature set to {t}"
        except ValueError:
            return "Invalid value. Usage: /temperature 0.7"


class TokensCommand(SlashCommand):
    name = "tokens"
    description = "Show token usage summary"

    def execute(self, args: str, repl: AgentREPL) -> str:
        s = repl.state
        return (
            f"Token usage:\n"
            f"  Input:  {s.total_input_tokens:,}\n"
            f"  Output: {s.total_output_tokens:,}\n"
            f"  Total:  {s.total_input_tokens + s.total_output_tokens:,}\n"
            f"  Messages in history: {s.message_count}"
        )


class ExitCommand(SlashCommand):
    name = "exit"
    description = "Exit the agent"

    def execute(self, args: str, repl: AgentREPL) -> str:
        raise SystemExit(0)


def create_default_registry() -> CommandRegistry:
    """Create registry with all built-in commands."""
    registry = CommandRegistry()
    for cmd_class in [
        HelpCommand,
        ClearCommand,
        ModelCommand,
        ProviderCommand,
        SystemCommand,
        TemperatureCommand,
        TokensCommand,
        ExitCommand,
    ]:
        registry.register(cmd_class())
    return registry
