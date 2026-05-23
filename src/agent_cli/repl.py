"""Core REPL loop - the heart of the agent CLI."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from agent_cli.config import Config
from agent_cli.state import ConversationState
from agent_cli.commands import CommandRegistry, create_default_registry


class AgentREPL:
    """Interactive REPL for the AI Agent CLI."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.state = ConversationState()
        self.registry: CommandRegistry = create_default_registry()
        self.console = Console()
        self.llm: BaseChatModel = self._build_llm()

    # --- LLM Management ---

    def _build_llm(self) -> BaseChatModel:
        """Create LLM instance from current config."""
        api_key = self.config.get_api_key()
        if self.config.provider == "openai":
            return ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                streaming=True,
            )
        elif self.config.provider == "anthropic":
            return ChatAnthropic(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
                streaming=True,
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def rebuild_llm(self) -> None:
        """Rebuild LLM after config change."""
        try:
            self.llm = self._build_llm()
        except ValueError as e:
            self.console.print(f"[red]Error rebuilding LLM: {e}[/red]")

    # --- Streaming Chat ---

    def chat(self, user_input: str) -> None:
        """Send message to LLM and stream response with Rich markdown rendering."""
        self.state.add_user_message(user_input)
        messages = self.state.get_messages_with_system(self.config.system_prompt)

        collected = []
        input_tokens = 0
        output_tokens = 0

        try:
            with Live("", console=self.console, refresh_per_second=8) as live:
                for chunk in self.llm.stream(messages):
                    if chunk.content:
                        collected.append(chunk.content)
                        full_text = "".join(collected)
                        live.update(Markdown(full_text))

                    # Extract token usage if available
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0) or input_tokens
                        output_tokens = getattr(usage, "output_tokens", 0) or output_tokens

            full_response = "".join(collected)
            self.state.add_ai_message(full_response)
            self.state.update_token_usage(input_tokens, output_tokens)

            if self.config.show_token_usage and (input_tokens or output_tokens):
                self.console.print(
                    f"[dim]tokens: in={input_tokens:,} out={output_tokens:,}[/dim]"
                )

        except Exception as e:
            # Remove the user message we just added since the call failed
            self.state.messages.pop()
            self.console.print(f"[red]Error: {e}[/red]")

    # --- REPL Loop ---

    def run(self) -> None:
        """Main REPL loop."""
        # Build completer for slash commands
        cmd_completer = WordCompleter(
            [f"/{name}" for name in self.registry.command_names],
            sentence=True,
        )
        session: PromptSession = PromptSession(
            history=InMemoryHistory(),
            completer=cmd_completer,
        )

        self._print_welcome()

        while True:
            try:
                user_input = session.prompt("\n❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input:
                continue

            # Try slash command first
            was_command, result = self.registry.parse_and_execute(user_input, self)
            if was_command:
                if result:
                    self.console.print(Panel(result, border_style="cyan"))
                continue

            # Otherwise, send to LLM
            self.chat(user_input)

    def _print_welcome(self) -> None:
        self.console.print(
            Panel(
                f"[bold]AI Agent CLI[/bold] v0.1.0\n"
                f"Provider: {self.config.provider} | Model: {self.config.model}\n"
                f"Type [cyan]/help[/cyan] for commands, [cyan]/exit[/cyan] to quit.",
                border_style="green",
            )
        )
