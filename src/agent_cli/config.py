"""Configuration management for AI Agent CLI."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Agent configuration with sensible defaults."""

    # LLM settings
    provider: str = "openai"  # "openai" or "anthropic"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096

    # System prompt
    system_prompt: str = (
        "You are a helpful AI assistant running in a CLI environment. "
        "Be concise and direct in your responses. "
        "Use markdown formatting when it helps readability."
    )

    # Display
    show_token_usage: bool = True

    # Provider-specific model defaults
    MODEL_DEFAULTS: dict = field(
        default_factory=lambda: {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
        }
    )

    def get_api_key(self) -> str:
        """Get API key for current provider."""
        if self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
        elif self.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        else:
            key = ""
        if not key:
            raise ValueError(
                f"API key not found. Set {'OPENAI_API_KEY' if self.provider == 'openai' else 'ANTHROPIC_API_KEY'} environment variable."
            )
        return key
