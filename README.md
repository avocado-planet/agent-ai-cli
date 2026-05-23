# AI Agent CLI

AI Agentの学習を目的とした、Claude Code風のCLIツール。

## Tech Stack
- Python + LangChain / LangGraph
- Rich (ターミナルUI) + prompt_toolkit (入力補完)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
export OPENAI_API_KEY="sk-..."
python -m agent_cli.main
# 作業終了時
deactivate
```

## Slash Commands
- `/help` — コマンド一覧
- `/model` — モデル切り替え
- `/provider` — プロバイダー切り替え (openai / anthropic)
- `/system` — システムプロンプト変更
- `/temperature` — 温度設定
- `/tokens` — トークン使用量表示
- `/clear` — 会話履歴クリア
- `/exit` — 終了
