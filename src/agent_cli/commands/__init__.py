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


# --- Phase 2 Commands ---


class SaveCommand(SlashCommand):
    name = "save"
    description = "Save session (e.g. /save my_session)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        from agent_cli.session import save_session

        if repl.state.message_count == 0:
            return "Nothing to save — conversation is empty."
        session_name = args.strip() or None
        try:
            path = save_session(repl.state, repl.config, name=session_name)
            return f"Session saved: {path}"
        except Exception as e:
            return f"Save failed: {e}"


class LoadCommand(SlashCommand):
    name = "load"
    description = "Load session (e.g. /load my_session)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        from agent_cli.session import load_session

        session_name = args.strip()
        if not session_name:
            return "Usage: /load <session_name>  (use /sessions to list)"

        try:
            state, config_dict = load_session(session_name)
        except FileNotFoundError as e:
            return str(e)
        except Exception as e:
            return f"Load failed: {e}"

        # Restore state
        repl.state = state

        # Restore config and rebuild LLM
        if config_dict:
            repl.config.provider = config_dict.get("provider", repl.config.provider)
            repl.config.model = config_dict.get("model", repl.config.model)
            repl.config.temperature = config_dict.get("temperature", repl.config.temperature)
            repl.config.max_tokens = config_dict.get("max_tokens", repl.config.max_tokens)
            repl.config.system_prompt = config_dict.get("system_prompt", repl.config.system_prompt)
            repl.rebuild_llm()

        return (
            f"Session loaded: {session_name}\n"
            f"  Messages: {state.message_count}\n"
            f"  Model: {repl.config.provider}/{repl.config.model}"
        )


class SessionsCommand(SlashCommand):
    name = "sessions"
    description = "List saved sessions"

    def execute(self, args: str, repl: AgentREPL) -> str:
        from agent_cli.session import list_sessions

        sessions = list_sessions()
        if not sessions:
            return "No saved sessions found."

        lines = ["Saved sessions:\n"]
        for s in sessions:
            saved = s["saved_at"][:19].replace("T", " ")  # trim to readable
            lines.append(
                f"  {s['name']:<20} {s['message_count']:>3} msgs  "
                f"{s['model']:<25} {saved}"
            )
        return "\n".join(lines)


class ExportCommand(SlashCommand):
    name = "export"
    description = "Export conversation as Markdown (e.g. /export chat.md)"

    def execute(self, args: str, repl: AgentREPL) -> str:
        from pathlib import Path
        from agent_cli.session import export_as_markdown

        if repl.state.message_count == 0:
            return "Nothing to export — conversation is empty."

        filename = args.strip() or "conversation.md"
        if not filename.endswith(".md"):
            filename += ".md"

        md_content = export_as_markdown(repl.state, repl.config)
        path = Path(filename)
        path.write_text(md_content, encoding="utf-8")
        return f"Conversation exported: {path.resolve()}"


class CompactCommand(SlashCommand):
    name = "compact"
    description = "Summarize conversation to reduce token usage"

    def execute(self, args: str, repl: AgentREPL) -> str:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        if repl.state.message_count < 4:
            return "Conversation too short to compact (need at least 4 messages)."

        # Build a summarization request
        summary_prompt = (
            "Summarize the following conversation concisely. "
            "Preserve key facts, decisions, and context that would be needed "
            "to continue the conversation. Respond with the summary only.\n\n"
        )
        conversation_text = []
        for msg in repl.state.messages:
            if isinstance(msg, HumanMessage):
                conversation_text.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                conversation_text.append(f"Assistant: {msg.content}")

        summary_messages = [
            SystemMessage(content="You are a helpful summarizer."),
            HumanMessage(content=summary_prompt + "\n".join(conversation_text)),
        ]

        try:
            response = repl.llm.invoke(summary_messages)
        except Exception as e:
            return f"Compact failed: {e}"

        old_count = repl.state.message_count
        repl.state.clear()

        # Insert the summary as context for future messages
        repl.state.add_user_message("[Previous conversation summary]")
        repl.state.add_ai_message(response.content)

        return (
            f"Compacted {old_count} messages → 2 (summary).\n"
            f"Summary:\n{response.content[:300]}{'...' if len(response.content) > 300 else ''}"
        )


class ConfigCommand(SlashCommand):
    name = "config"
    description = "Show all current settings"

    def execute(self, args: str, repl: AgentREPL) -> str:
        c = repl.config
        s = repl.state
        return (
            f"Current Configuration:\n"
            f"  provider:      {c.provider}\n"
            f"  model:         {c.model}\n"
            f"  temperature:   {c.temperature}\n"
            f"  max_tokens:    {c.max_tokens}\n"
            f"  token_usage:   {c.show_token_usage}\n"
            f"  messages:      {s.message_count}\n"
            f"  total_tokens:  {s.total_input_tokens + s.total_output_tokens:,}\n"
            f"\nSystem prompt:\n  {c.system_prompt[:100]}{'...' if len(c.system_prompt) > 100 else ''}"
        )


def create_default_registry() -> CommandRegistry:
    """Create registry with all built-in commands."""
    registry = CommandRegistry()
    for cmd_class in [
        # Phase 1
        HelpCommand,
        ClearCommand,
        ModelCommand,
        ProviderCommand,
        SystemCommand,
        TemperatureCommand,
        TokensCommand,
        ExitCommand,
        # Phase 2
        SaveCommand,
        LoadCommand,
        SessionsCommand,
        ExportCommand,
        CompactCommand,
        ConfigCommand,
    ]:
        registry.register(cmd_class())
    return registry
