"""Entry point for the AI Agent CLI."""

import sys
from agent_cli.config import Config
from agent_cli.repl import AgentREPL


def main() -> None:
    config = Config()

    # Allow quick provider override via CLI arg
    if len(sys.argv) > 1:
        provider = sys.argv[1].lower()
        if provider in ("openai", "anthropic"):
            config.provider = provider
            config.model = config.MODEL_DEFAULTS[provider]

    try:
        config.get_api_key()  # Validate early
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    repl = AgentREPL(config)
    repl.run()


if __name__ == "__main__":
    main()
