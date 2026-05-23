"""Conversation state management."""

from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


@dataclass
class ConversationState:
    """Holds conversation history and metadata."""

    messages: list[BaseMessage] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        self.messages.append(AIMessage(content=content))

    def get_messages_with_system(self, system_prompt: str) -> list[BaseMessage]:
        """Return messages list with system prompt prepended."""
        return [SystemMessage(content=system_prompt)] + self.messages

    def clear(self) -> None:
        """Clear conversation history (keep token counts)."""
        self.messages.clear()

    def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    @property
    def message_count(self) -> int:
        return len(self.messages)
