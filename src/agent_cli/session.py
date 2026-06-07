"""Session persistence — serialize/deserialize conversation state to JSON."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from agent_cli.state import ConversationState
from agent_cli.config import Config

# Default directory for saved sessions
DEFAULT_SESSIONS_DIR = Path.home() / ".ai-agent-cli" / "sessions"


def _message_to_dict(msg: BaseMessage) -> dict:
    """Convert a LangChain message to a serializable dict."""
    type_map = {
        HumanMessage: "human",
        AIMessage: "ai",
        SystemMessage: "system",
    }
    return {
        "type": type_map.get(type(msg), "unknown"),
        "content": msg.content,
    }


def _dict_to_message(d: dict) -> BaseMessage:
    """Reconstruct a LangChain message from a dict."""
    type_map = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }
    cls = type_map.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown message type: {d['type']}")
    return cls(content=d["content"])


def save_session(
    state: ConversationState,
    config: Config,
    name: str | None = None,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
) -> Path:
    """
    Save conversation state and config to a JSON file.
    Returns the path of the saved file.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)

    if not name:
        name = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    filepath = sessions_dir / f"{safe_name}.json"

    data = {
        "version": 1,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "provider": config.provider,
            "model": config.model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "system_prompt": config.system_prompt,
        },
        "token_usage": {
            "total_input": state.total_input_tokens,
            "total_output": state.total_output_tokens,
        },
        "messages": [_message_to_dict(m) for m in state.messages],
    }

    filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return filepath


def load_session(
    name: str,
    sessions_dir: Path = DEFAULT_SESSIONS_DIR,
) -> tuple[ConversationState, dict]:
    """
    Load a session from JSON.
    Returns (ConversationState, config_dict).
    """
    filepath = sessions_dir / f"{name}.json"
    if not filepath.exists():
        # Try with sanitized name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        filepath = sessions_dir / f"{safe_name}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Session not found: {name}")

    data = json.loads(filepath.read_text())

    state = ConversationState()
    state.messages = [_dict_to_message(m) for m in data.get("messages", [])]
    token_usage = data.get("token_usage", {})
    state.total_input_tokens = token_usage.get("total_input", 0)
    state.total_output_tokens = token_usage.get("total_output", 0)

    return state, data.get("config", {})


def list_sessions(sessions_dir: Path = DEFAULT_SESSIONS_DIR) -> list[dict]:
    """
    List all saved sessions with metadata.
    Returns list of dicts with name, saved_at, message_count, model.
    """
    if not sessions_dir.exists():
        return []

    sessions = []
    for f in sorted(sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            sessions.append({
                "name": f.stem,
                "saved_at": data.get("saved_at", "unknown"),
                "message_count": len(data.get("messages", [])),
                "model": data.get("config", {}).get("model", "unknown"),
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions


def export_as_markdown(state: ConversationState, config: Config) -> str:
    """Export the conversation as a Markdown string."""
    lines = [
        "# AI Agent CLI — Conversation Export",
        "",
        f"- **Provider**: {config.provider}",
        f"- **Model**: {config.model}",
        f"- **Tokens**: input={state.total_input_tokens:,} / output={state.total_output_tokens:,}",
        f"- **Messages**: {state.message_count}",
        "",
        "---",
        "",
    ]

    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            lines.append("## 🧑 User")
            lines.append("")
            lines.append(msg.content)
        elif isinstance(msg, AIMessage):
            lines.append("## 🤖 Assistant")
            lines.append("")
            lines.append(msg.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
