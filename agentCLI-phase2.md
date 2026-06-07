# AI Agent CLI — Phase 2 技術解説ガイド

Phase 2で追加した「セッション永続化・会話エクスポート・会話圧縮」の技術要素を、
ソースコードと対応させて解説する学習資料です。

---

## 目次

1. [Phase 2 で何が変わったか](#1-phase-2-で何が変わったか)
2. [セッション永続化 — シリアライズの設計](#2-セッション永続化--シリアライズの設計)
3. [LangChainメッセージのシリアライズ問題](#3-langchainメッセージのシリアライズ問題)
4. [/compact — LLMを使った会話圧縮](#4-compact--llmを使った会話圧縮)
5. [/export — Markdown生成パターン](#5-export--markdown生成パターン)
6. [コマンドの遅延import戦略](#6-コマンドの遅延import戦略)
7. [ファイルシステム操作のベストプラクティス](#7-ファイルシステム操作のベストプラクティス)
8. [Phase 1 → Phase 2 の差分に見る拡張性](#8-phase-1--phase-2-の差分に見る拡張性)
9. [Phase 3 への接続ポイント](#9-phase-3-への接続ポイント)

---

## 1. Phase 2 で何が変わったか

### 追加ファイル

| ファイル | 役割 |
|---------|------|
| `session.py` | セッションのJSON保存・読み込み・一覧・Markdownエクスポート |

### 追加コマンド（6個、合計14個に）

| コマンド | 機能 | カテゴリ |
|---------|------|---------|
| `/save [name]` | セッションをJSONファイルに保存 | 永続化 |
| `/load <name>` | 保存済みセッションを復元 | 永続化 |
| `/sessions` | 保存済みセッション一覧 | 永続化 |
| `/export [file.md]` | 会話をMarkdownファイルに出力 | エクスポート |
| `/compact` | 会話をLLMで要約してトークン削減 | 最適化 |
| `/config` | 全設定の一覧表示 | 情報表示 |

### 変更されなかったファイル

`config.py`、`state.py`、`repl.py`、`main.py` は **一切変更なし**。
Phase 1の設計が正しかったことの証左。新機能はすべて「追加」のみで実現。

---

## 2. セッション永続化 — シリアライズの設計

### 保存ファイルの構造

```json
{
  "version": 1,
  "name": "my_session",
  "saved_at": "2024-12-15T10:30:00+00:00",
  "config": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 4096,
    "system_prompt": "You are a helpful..."
  },
  "token_usage": {
    "total_input": 1500,
    "total_output": 800
  },
  "messages": [
    {"type": "human", "content": "Hello"},
    {"type": "ai", "content": "Hi there!"}
  ]
}
```

### 設計上の重要な判断

**`version` フィールド:**

```python
data = {
    "version": 1,  # ← スキーマバージョン
    ...
}
```

将来フォーマットを変更したとき（例: ツール呼び出し履歴の追加）に、
古いファイルと新しいファイルを区別してマイグレーションできる。
永続化を含むシステムでは初期段階から入れておくべきフィールド。

**会話とConfigの両方を保存する理由:**

`/load` で復元するとき、モデルやtemperatureも元に戻す。
「この会話はgpt-4oのtemp=0.3で進めていた」という文脈が失われると、
復元後の応答品質が変わってしまう。

**保存場所の設計:**

```python
DEFAULT_SESSIONS_DIR = Path.home() / ".ai-agent-cli" / "sessions"
#                      ~/.ai-agent-cli/sessions/
```

ホームディレクトリ直下のドットフォルダ（隠しフォルダ）にする慣例は
Unix系ツールの標準的なパターン。`.gitconfig`、`.ssh` と同じ考え方。

---

## 3. LangChainメッセージのシリアライズ問題

### なぜ独自のシリアライズが必要か

LangChainの `BaseMessage` はそのまま `json.dumps()` できない:

```python
msg = HumanMessage(content="Hello")
json.dumps(msg)  # → TypeError: Object of type HumanMessage is not JSON serializable
```

そこで、変換関数を自前で用意する:

```python
def _message_to_dict(msg: BaseMessage) -> dict:
    """LangChainメッセージ → シリアライズ可能なdict"""
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
    """dict → LangChainメッセージに復元"""
    type_map = {
        "human": HumanMessage,
        "ai": AIMessage,
        "system": SystemMessage,
    }
    cls = type_map.get(d["type"])
    if cls is None:
        raise ValueError(f"Unknown message type: {d['type']}")
    return cls(content=d["content"])
```

### 技術ポイント: 双方向マッピング

`_message_to_dict` は **クラス → 文字列**、`_dict_to_message` は **文字列 → クラス** の変換。
この双方向マッピングは永続化の基本パターンで、ORMやProtobufでも同じ考え方が使われる。

### なぜ `type(msg)` でマッピングするのか

`isinstance()` ではなく `type()` を使っている理由:

```python
# isinstance だと継承関係で曖昧になる可能性がある
isinstance(HumanMessage("hi"), BaseMessage)  # True — どの型かわからない

# type() なら正確にクラスを特定できる
type(HumanMessage("hi")) == HumanMessage  # True
```

### Phase 3での拡張予測

Phase 3でツール呼び出しが入ると、`ToolMessage` 型が追加される。
このとき必要な変更は `type_map` に1行追加するだけ:

```python
type_map = {
    HumanMessage: "human",
    AIMessage: "ai",
    SystemMessage: "system",
    ToolMessage: "tool",      # ← 追加
}
```

---

## 4. /compact — LLMを使った会話圧縮

### なぜ圧縮が必要か

LLMのコンテキストウィンドウには上限がある（例: gpt-4o-miniは128Kトークン）。
長い会話を続けると:

1. **コスト増**: 毎回の入力トークンが増え続ける
2. **速度低下**: 入力が長いほど応答が遅くなる
3. **コンテキスト溢れ**: 上限を超えるとエラーになる

`/compact` はLLM自身に会話を要約させ、履歴を圧縮する。

### 実装の流れ

```python
def execute(self, args: str, repl: AgentREPL) -> str:
    # 1. 会話が短すぎたら何もしない
    if repl.state.message_count < 4:
        return "Conversation too short to compact."

    # 2. 要約用のプロンプトを組み立てる
    summary_prompt = (
        "Summarize the following conversation concisely. "
        "Preserve key facts, decisions, and context..."
    )
    conversation_text = []
    for msg in repl.state.messages:
        if isinstance(msg, HumanMessage):
            conversation_text.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            conversation_text.append(f"Assistant: {msg.content}")

    # 3. 要約をLLMに依頼（通常の会話とは別のメッセージで）
    summary_messages = [
        SystemMessage(content="You are a helpful summarizer."),
        HumanMessage(content=summary_prompt + "\n".join(conversation_text)),
    ]
    response = repl.llm.invoke(summary_messages)

    # 4. 元の履歴をクリアし、要約を2メッセージとして挿入
    old_count = repl.state.message_count
    repl.state.clear()
    repl.state.add_user_message("[Previous conversation summary]")
    repl.state.add_ai_message(response.content)
```

### 技術ポイント

**要約を通常の会話履歴とは別コンテキストで実行:**

```python
summary_messages = [
    SystemMessage(content="You are a helpful summarizer."),  # ← 専用のシステムプロンプト
    HumanMessage(content=summary_prompt + ...),
]
response = repl.llm.invoke(summary_messages)  # ← 会話履歴を含まない
```

もし `repl.state.messages` に要約リクエストを追加してしまうと、
要約文がユーザーの発言として履歴に残り、会話が壊れる。

**要約結果を `HumanMessage` + `AIMessage` のペアとして挿入:**

```python
repl.state.add_user_message("[Previous conversation summary]")
repl.state.add_ai_message(response.content)
```

LLMは `User → AI → User → AI` の交互パターンを期待する。
要約をAIメッセージだけにすると、直前のユーザーメッセージが存在せず
APIエラーになるプロバイダーがある。ペアで入れるのが安全。

**Claude Code との比較:**

Claude Code にも同様の `/compact` コマンドがあり、同じパターン
（LLMによる自動要約 → 履歴置換）を採用している。
プロダクション品質のAgent CLIでは必須の機能。

---

## 5. /export — Markdown生成パターン

### 実装

```python
def export_as_markdown(state: ConversationState, config: Config) -> str:
    lines = [
        "# AI Agent CLI — Conversation Export",
        "",
        f"- **Provider**: {config.provider}",
        f"- **Model**: {config.model}",
        ...
    ]

    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            lines.append("## 🧑 User")
            lines.append("")
            lines.append(msg.content)
        elif isinstance(msg, AIMessage):
            lines.append("## 🤖 Assistant")
            ...

    return "\n".join(lines)
```

### 技術ポイント: リストに溜めて最後に結合

```python
lines = []
lines.append("...")
lines.append("...")
return "\n".join(lines)
```

`+=` で文字列を連結するよりも、リストに `append` して最後に `join` する方が
パフォーマンスが良い。Pythonの文字列はイミュータブルなので、
`s += "text"` は毎回新しい文字列オブジェクトを生成する。
`list.append` + `join` なら生成は1回だけ。

### ファイル名の自動補完

```python
# ExportCommand.execute()
filename = args.strip() or "conversation.md"    # デフォルト名
if not filename.endswith(".md"):
    filename += ".md"                            # 拡張子の自動付与
```

ユーザーが `/export chat` と入力しても `/export chat.md` と入力しても
同じ `chat.md` が生成される。ユーザーに「.mdを付けてください」と言わせない。

---

## 6. コマンドの遅延import戦略

### Phase 2コマンドのimportパターン

```python
class SaveCommand(SlashCommand):
    def execute(self, args: str, repl: AgentREPL) -> str:
        from agent_cli.session import save_session  # ← 関数の中でimport
        ...

class CompactCommand(SlashCommand):
    def execute(self, args: str, repl: AgentREPL) -> str:
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # ← 関数の中
        ...
```

### なぜファイル先頭ではなくメソッド内でimportするのか

**遅延import (lazy import)** と呼ばれるテクニックで、3つの利点がある:

1. **起動速度**: `session.py` は `/save` を初めて使うまで読み込まれない。
   CLIの起動が速くなる。

2. **循環import回避**: `commands/__init__.py` が `session.py` をトップレベルで
   importすると、将来 `session.py` が他のモジュールを参照したときに循環が起きやすい。

3. **依存の局所化**: `SaveCommand` の実装を見れば、何に依存しているかが
   そのメソッド内で完結する。ファイル先頭のimport群から探す必要がない。

### トレードオフ

- メソッドが呼ばれるたびに `import` 文が実行されるが、
  Python は一度importしたモジュールを `sys.modules` にキャッシュするので、
  **2回目以降はほぼゼロコスト**。

---

## 7. ファイルシステム操作のベストプラクティス

### pathlib の活用

`session.py` では `os.path` ではなく `pathlib.Path` を使っている:

```python
from pathlib import Path

DEFAULT_SESSIONS_DIR = Path.home() / ".ai-agent-cli" / "sessions"
#                       ↑ ホームdir    ↑ / 演算子でパス結合
```

`pathlib` の利点:
- `/` 演算子でパス結合（`os.path.join()` より読みやすい）
- `.exists()`, `.read_text()`, `.write_text()` が直接使える
- OS間のパス区切り文字を自動処理

### ディレクトリの安全な作成

```python
sessions_dir.mkdir(parents=True, exist_ok=True)
#                   ↑ 親ディレクトリも作る  ↑ 既に存在してもエラーにしない
```

これは `mkdir -p` と同等。`exist_ok=False`（デフォルト）だと
ディレクトリが既にある場合に `FileExistsError` が発生する。

### ファイル名のサニタイズ

```python
safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
```

ユーザーが `/save my session!` と入力した場合、
ファイル名に使えない文字（スペース、`!`）を `_` に置換して `my_session_` にする。
ジェネレータ式を `"".join()` に渡すパターン。

### 一覧取得時のソート

```python
for f in sorted(sessions_dir.glob("*.json"), key=os.path.getmtime, reverse=True):
```

- `glob("*.json")` — ディレクトリ内の全JSONファイルを取得
- `key=os.path.getmtime` — ファイルの最終更新時刻でソート
- `reverse=True` — 新しい順（降順）

### 読み込み時のエラー耐性

```python
for f in ...:
    try:
        data = json.loads(f.read_text())
        sessions.append({...})
    except (json.JSONDecodeError, KeyError):
        continue  # ← 壊れたファイルはスキップして続行
```

ユーザーが手動でJSONを編集して壊す可能性がある。
一覧表示が1ファイルの破損で全体停止しないように `continue` で飛ばす。

---

## 8. Phase 1 → Phase 2 の差分に見る拡張性

### 変更点のまとめ

```
変更ファイル:
  commands/__init__.py  — 6つの新コマンドクラスを追加、registryに登録

新規ファイル:
  session.py           — セッション永続化ロジック

変更なし:
  config.py            — そのまま
  state.py             — そのまま
  repl.py              — そのまま
  main.py              — そのまま
```

### Open-Closed Principle の実証

Phase 1 のガイドで述べた「新しいコマンドを追加するときに既存コードを変更しない」
という原則が、Phase 2 で実証された:

1. `session.py`（新規）— 永続化ロジックは独立モジュール
2. `commands/__init__.py`（追記のみ）— 新コマンドクラスの定義とregistryへの登録
3. `repl.py`（変更なし）— REPLループは何も知らなくてよい

`repl.py` が新コマンドの存在を **一切知らない** のが重要。
`registry.parse_and_execute()` がディスパッチするので、
REPL側のコードは `/save` が追加されても `/compact` が追加されても変わらない。

### 責務の分離が生む保守性

もし `/save` にバグがあったとき、修正するのは:
- `SaveCommand.execute()` か `session.save_session()` のどちらか
- `repl.py` や `state.py` を触る必要は絶対にない

これがモジュール分割の実践的なメリット。

---

## 9. Phase 3 への接続ポイント

### Phase 3（LangGraphエージェント化）で変わる部分

Phase 3 では `repl.py` の `chat()` メソッドを LangGraph の `StateGraph` に置き換える。
Phase 2 で追加したモジュールへの影響:

| モジュール | 影響 |
|-----------|------|
| `session.py` | `ToolMessage` のシリアライズを `type_map` に追加するだけ |
| `commands/__init__.py` | Phase 2のコマンドは変更不要 |
| `/compact` | ツール呼び出し履歴も要約対象に含める調整が必要 |
| `/export` | `ToolMessage` の表示フォーマットを追加 |

### /compact の進化予測

Phase 3 以降、会話にはツール実行のログも含まれる:

```
User: このディレクトリのファイル一覧を見せて
AI: [bash_execute ツールを呼び出し]
Tool: total 12\n-rw-r--r-- 1 user user 1234 ...
AI: 以下のファイルがあります: ...
```

ツール実行の詳細ログは要約時に重要度が低いので、
圧縮率を上げるために「ツール呼び出しの結果は要約で省略」という
戦略が有効になる。

---

## まとめ: Phase 2 で学んだ技術要素チェックリスト

- [ ] JSONによるセッション永続化（スキーマバージョニング含む）
- [ ] LangChainメッセージの双方向シリアライズ（`type_map` パターン）
- [ ] LLMを使った会話圧縮（`/compact`）とそのプロンプト設計
- [ ] 要約を別コンテキストで実行する理由
- [ ] `pathlib.Path` によるファイルシステム操作
- [ ] ファイル名サニタイズ
- [ ] 遅延import（lazy import）の利点とトレードオフ
- [ ] Markdown生成のリスト+join パターン
- [ ] Open-Closed Principle の実証（既存ファイル変更なし）
- [ ] エラー耐性のある一覧取得（壊れたファイルのスキップ）
