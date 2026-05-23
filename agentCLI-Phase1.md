# AI Agent CLI — Phase 1

Phase 1で構築した「REPL + ストリーミングLLM + スラッシュコマンド」の全コードを、
技術要素ごとに分解して解説する学習資料です。

---

## 目次

1. [全体アーキテクチャ](#1-全体アーキテクチャ)
2. [REPLパターン — CLIエージェントの心臓部](#2-replパターン--cliエージェントの心臓部)
3. [LangChainのメッセージモデル](#3-langchainのメッセージモデル)
4. [LLMプロバイダー抽象化 — BaseChatModel](#4-llmプロバイダー抽象化--basechatmodel)
5. [ストリーミング応答 — stream() + Rich Live](#5-ストリーミング応答--stream--rich-live)
6. [スラッシュコマンドシステム — Registryパターン](#6-スラッシュコマンドシステム--registryパターン)
7. [Pythonの設計テクニック集](#7-pythonの設計テクニック集)
8. [依存ライブラリの役割整理](#8-依存ライブラリの役割整理)
9. [Phase 2以降への接続ポイント](#9-phase-2以降への接続ポイント)

---

## 1. 全体アーキテクチャ

### データフロー

```
ユーザー入力
    │
    ▼
┌─────────────────────────┐
│  prompt_toolkit (REPL)  │  ← 入力補完・履歴
│    PromptSession        │
└────────┬────────────────┘
         │
         ▼
   "/" で始まる？ ─── Yes ──→ CommandRegistry.parse_and_execute()
         │                         │
         No                        ▼
         │                   SlashCommand.execute()
         ▼                         │
   ConversationState               ▼
   .add_user_message()       結果をPanelで表示
         │
         ▼
   BaseChatModel.stream()  ← LangChain統一インターフェース
         │
         ▼
   Rich Live + Markdown    ← リアルタイムレンダリング
         │
         ▼
   ConversationState
   .add_ai_message()
   .update_token_usage()
```

### ファイルの責務

| ファイル | 責務 | 主要クラス |
|---------|------|-----------|
| `config.py` | 設定の集約・APIキー管理 | `Config` |
| `state.py` | 会話履歴・トークン使用量 | `ConversationState` |
| `commands/__init__.py` | コマンド基盤＋組み込みコマンド | `SlashCommand`, `CommandRegistry` |
| `repl.py` | REPL本体・LLM管理・ストリーミング | `AgentREPL` |
| `main.py` | エントリーポイント | `main()` |

**設計原則**: 各ファイルが**単一の責務**を持つ。`repl.py`は「つなぎ役」として他モジュールを組み合わせるオーケストレーターの役割。

---

## 2. REPLパターン — CLIエージェントの心臓部

### REPLとは

**R**ead → **E**valuate → **P**rint → **L**oop の略。対話型プログラムの基本パターン。

```python
# repl.py の run() メソッド — 最もシンプルな形に還元すると：
while True:
    user_input = input("❯ ")    # Read
    response = llm(user_input)   # Evaluate
    print(response)              # Print
    # ← Loop (whileの先頭に戻る)
```

### 実際のコードが加えている工夫

```python
def run(self) -> None:
    session: PromptSession = PromptSession(
        history=InMemoryHistory(),        # ← 上下キーで入力履歴
        completer=cmd_completer,          # ← Tabキーで/コマンド補完
    )

    while True:
        try:
            user_input = session.prompt("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):  # ← Ctrl+C / Ctrl+D の安全な処理
            break

        if not user_input:
            continue  # ← 空行はスキップ

        # コマンド分岐
        was_command, result = self.registry.parse_and_execute(user_input, self)
        if was_command:
            if result:
                self.console.print(Panel(result, border_style="cyan"))
            continue  # ← コマンド処理後はLLMに送らない

        self.chat(user_input)  # ← LLM呼び出し
```

**重要ポイント**:
- `input()` ではなく `prompt_toolkit.PromptSession` を使う理由は、補完・履歴・キーバインドなどターミナルUXの基盤機能が得られるため
- `EOFError`（Ctrl+D）と `KeyboardInterrupt`（Ctrl+C）の両方をキャッチして安全に終了
- スラッシュコマンドは **LLM呼び出しの前に** インターセプトされる（`continue`で早期リターン）

---

## 3. LangChainのメッセージモデル

### ChatモデルのAPI形式

LLMのChat APIは「メッセージの配列」を受け取る。LangChainはこれをPythonクラスで表現する。

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
```

| クラス | 役割 | OpenAI APIでの対応 |
|--------|------|-------------------|
| `SystemMessage` | LLMの振る舞いを指示 | `{"role": "system", ...}` |
| `HumanMessage` | ユーザーの発言 | `{"role": "user", ...}` |
| `AIMessage` | LLMの応答 | `{"role": "assistant", ...}` |

### state.py での会話管理

```python
@dataclass
class ConversationState:
    messages: list[BaseMessage] = field(default_factory=list)

    def get_messages_with_system(self, system_prompt: str) -> list[BaseMessage]:
        """毎回のAPI呼び出し時にシステムプロンプトを先頭に付与"""
        return [SystemMessage(content=system_prompt)] + self.messages
```

**なぜシステムプロンプトを `messages` に含めず毎回先頭に付けるのか？**

1. `/system` コマンドでプロンプトを変更したとき、即座に反映される
2. `/clear` で会話履歴をクリアしてもシステムプロンプトは消えない
3. システムプロンプトは「設定」であり「会話の一部」ではない

### トークンの累積追跡

```python
def update_token_usage(self, input_tokens: int, output_tokens: int) -> None:
    self.total_input_tokens += input_tokens   # ← 累積加算
    self.total_output_tokens += output_tokens
```

`clear()` で会話履歴を消しても **トークン使用量はリセットしない**。
セッション全体のコスト把握のため。

---

## 4. LLMプロバイダー抽象化 — BaseChatModel

### ポリモーフィズムによるプロバイダー切り替え

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
```

`ChatOpenAI` と `ChatAnthropic` はどちらも `BaseChatModel` を継承しており、
同じインターフェース（`.invoke()`, `.stream()`, `.bind_tools()` 等）を持つ。

```python
class AgentREPL:
    def __init__(self, ...):
        self.llm: BaseChatModel = self._build_llm()  # ← 型は基底クラス

    def _build_llm(self) -> BaseChatModel:
        if self.config.provider == "openai":
            return ChatOpenAI(...)    # ← 具象クラスA
        elif self.config.provider == "anthropic":
            return ChatAnthropic(...) # ← 具象クラスB
```

**これが意味すること**: `chat()` メソッド内の `self.llm.stream(messages)` は、
OpenAIでもAnthropicでも **コードを変えずに** 動く。

### rebuild_llm() パターン

```python
def rebuild_llm(self) -> None:
    """設定変更後にLLMインスタンスを再構築"""
    try:
        self.llm = self._build_llm()
    except ValueError as e:
        self.console.print(f"[red]Error rebuilding LLM: {e}[/red]")
```

`/model gpt-4o` や `/temperature 0.3` 実行時に呼ばれる。
LangChainのChatモデルは **イミュータブル**（作成後にパラメータ変更不可）なので、
設定を変えたら新しいインスタンスを作り直す必要がある。

---

## 5. ストリーミング応答 — stream() + Rich Live

### なぜストリーミングが必要か

LLMの応答は数秒〜数十秒かかる。`.invoke()` だと全文完成まで何も表示されない。
`.stream()` を使うと、生成されたトークンが **逐次届く** ので、タイプしているように表示できる。

### ストリーミングの仕組み

```python
def chat(self, user_input: str) -> None:
    self.state.add_user_message(user_input)
    messages = self.state.get_messages_with_system(self.config.system_prompt)

    collected = []  # ← 全チャンクを蓄積

    with Live("", console=self.console, refresh_per_second=8) as live:
        for chunk in self.llm.stream(messages):
            if chunk.content:
                collected.append(chunk.content)
                full_text = "".join(collected)        # ← 毎回全文を結合
                live.update(Markdown(full_text))      # ← Markdownとして再レンダリング
```

### 技術的なポイント

**`stream()` が返す `chunk` の構造:**

```python
# 各chunkはAIMessageChunkオブジェクト
# chunk.content = "Hello"  （数トークン分のテキスト断片）
# chunk.usage_metadata = UsageMetadata(input_tokens=..., output_tokens=...)
#   ↑ 最後のchunkにのみ含まれることが多い
```

**Rich Liveの動作原理:**

```python
with Live("", console=self.console, refresh_per_second=8) as live:
    # Live は refresh_per_second の頻度でターミナルを「再描画」する
    # live.update() で表示内容を差し替えると、
    # 前の表示が消えて新しい内容に置き換わる（フリッカーなし）
    live.update(Markdown(full_text))
```

`refresh_per_second=8` は「1秒あたり8回まで画面を更新する」制限。
チャンクは毎秒数十個届くが、描画は8回に間引かれるのでターミナルがちらつかない。

**なぜ毎回 `"".join(collected)` で全文を結合するのか？**

Markdownの表（テーブル）やコードブロックは、途中の状態では正しくパースできない。
毎回全文をMarkdownとしてパースし直すことで、構文が閉じた瞬間に正しくレンダリングされる。

### エラー時のロールバック

```python
except Exception as e:
    self.state.messages.pop()  # ← 追加したユーザーメッセージを取り消す
    self.console.print(f"[red]Error: {e}[/red]")
```

API呼び出しが失敗した場合、`add_user_message()` で追加したメッセージを消す。
こうしないと、次の呼び出し時に「応答のないユーザーメッセージ」が履歴に残り、
LLMが混乱する。

---

## 6. スラッシュコマンドシステム — Registryパターン

### 設計パターン: Command + Registry

このコマンドシステムは2つのGoFデザインパターンを組み合わせている。

**Commandパターン**: 操作を独立したオブジェクトとしてカプセル化

```python
class SlashCommand(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def execute(self, args: str, repl: AgentREPL) -> str | None: ...
```

**Registryパターン**: コマンドを名前で引ける辞書に登録

```python
class CommandRegistry:
    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}

    def register(self, cmd: SlashCommand) -> None:
        self._commands[cmd.name] = cmd  # ← 名前をキーに登録

    def get(self, name: str) -> SlashCommand | None:
        return self._commands.get(name)  # ← 名前で引く
```

### なぜこの設計が優れているか

**新しいコマンドを追加するときの手順:**

1. `SlashCommand` を継承した新クラスを書く
2. `create_default_registry()` のリストに追加する
3. **以上。** `repl.py` や他のコマンドには一切触れない

これが **Open-Closed Principle**（拡張に対して開き、修正に対して閉じている）の実践。

### コマンドディスパッチの流れ

```python
def parse_and_execute(self, user_input: str, repl: AgentREPL) -> tuple[bool, str | None]:
    # 1. "/" で始まらなければコマンドではない
    if not user_input.startswith("/"):
        return False, None

    # 2. コマンド名と引数に分割
    parts = user_input[1:].split(maxsplit=1)  # "/model gpt-4o" → ["model", "gpt-4o"]
    cmd_name = parts[0].lower()               # "model"
    args = parts[1] if len(parts) > 1 else "" # "gpt-4o"

    # 3. 登録済みコマンドから検索
    cmd = self.get(cmd_name)
    if cmd is None:
        return True, f"Unknown command: /{cmd_name}..."

    # 4. 実行
    result = cmd.execute(args, repl)
    return True, result
```

**戻り値 `tuple[bool, str | None]` の設計意図:**
- `(False, None)` — コマンドではない → REPLはLLMに送る
- `(True, "...")` — コマンドだった → 結果を表示
- `(True, None)` — コマンドだったが表示すべきメッセージなし

### TYPE_CHECKING による循環import回避

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_cli.repl import AgentREPL
```

`commands/__init__.py` は `AgentREPL` の型アノテーションに使うが、
実行時にimportすると `repl.py ↔ commands` の循環importになる。
`TYPE_CHECKING` は **型チェッカー（mypy等）実行時のみ `True`** になる定数で、
実行時にはimportされない。

---

## 7. Pythonの設計テクニック集

### `@dataclass` — ボイラープレート削減

```python
@dataclass
class Config:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
```

`__init__`、`__repr__`、`__eq__` が自動生成される。設定やステートの保持に最適。

### `field(default_factory=...)` — ミュータブルなデフォルト値

```python
messages: list[BaseMessage] = field(default_factory=list)
MODEL_DEFAULTS: dict = field(default_factory=lambda: {"openai": "gpt-4o-mini", ...})
```

Pythonでは `messages: list = []` とすると全インスタンスが同じリストオブジェクトを共有してしまう。
`default_factory` は **インスタンスごとに新しいオブジェクト** を生成する。

### `from __future__ import annotations` — 遅延アノテーション評価

```python
from __future__ import annotations  # ← ファイル先頭

class AgentREPL:
    def __init__(self, config: Config | None = None) -> None:  # ← 3.10未満でも `|` が使える
```

型アノテーションを文字列として扱い、実行時には評価しない。
`Config | None` のような新しい構文を古いPythonでも使える。

### `getattr()` による安全な属性アクセス

```python
usage = getattr(chunk, "usage_metadata", None)
if usage:
    input_tokens = getattr(usage, "input_tokens", 0) or input_tokens
```

ストリーミングの各チャンクに `usage_metadata` が含まれるかはプロバイダーによる。
`getattr(obj, "attr", default)` は属性が存在しなければ `default` を返すので、
`AttributeError` を避けられる。

### `raise SystemExit(0)` — 例外によるフロー制御

```python
class ExitCommand(SlashCommand):
    def execute(self, args, repl):
        raise SystemExit(0)  # ← プロセスを終了
```

`/exit` コマンドがREPLループの深い場所から `sys.exit()` 相当の終了を実行する。
`SystemExit` は `BaseException` のサブクラスなので `except Exception` では捕捉されず、
REPLの `while True` を安全に抜ける。

---

## 8. 依存ライブラリの役割整理

### LangChain系

| パッケージ | 役割 | 本プロジェクトでの使い方 |
|-----------|------|----------------------|
| `langchain-core` | メッセージ型、基底クラス | `BaseMessage`, `HumanMessage`, `BaseChatModel` |
| `langchain-openai` | OpenAI接続 | `ChatOpenAI` |
| `langchain-anthropic` | Anthropic接続 | `ChatAnthropic` |

**LangChainを使う最大のメリット**: `.stream()` や `.invoke()` が **プロバイダーを問わず同じインターフェース** で使える。直接REST APIを叩く場合、OpenAIとAnthropicではリクエスト形式・ストリーミング形式・トークン使用量の取得方法がすべて異なる。

### ターミナルUI系

| パッケージ | 役割 | 本プロジェクトでの使い方 |
|-----------|------|----------------------|
| `rich` | ターミナル装飾 | `Console`, `Panel`, `Markdown`, `Live` |
| `prompt_toolkit` | 高機能入力 | `PromptSession`, `WordCompleter`, `InMemoryHistory` |

**rich と prompt_toolkit の使い分け:**
- `prompt_toolkit` = **入力** を担当（Tab補完、入力履歴、キーバインド）
- `rich` = **出力** を担当（色付き表示、Markdown描画、パネル、ストリーミング表示）

---

## 9. Phase 2以降への接続ポイント

Phase 1のコードには、今後の拡張を意識した「フック」が埋め込まれている。

### → Phase 2: コマンド拡充

`CommandRegistry` が拡張可能なので、新コマンドはクラスを追加するだけ:

```python
class SaveCommand(SlashCommand):
    name = "save"
    description = "Save conversation to file"

    def execute(self, args, repl):
        import json
        # repl.state.messages をシリアライズ
        ...
```

### → Phase 3: LangGraphエージェント化

現在の `chat()` メソッドを LangGraph の `StateGraph` に置き換える:

```python
# 現在（Phase 1）
def chat(self, user_input):
    response = self.llm.stream(messages)  # ← 単純なLLM呼び出し

# Phase 3 では
def chat(self, user_input):
    result = self.graph.invoke({"messages": messages})  # ← グラフ実行
    # graph内でLLM → ツール判断 → ツール実行 → LLM のループが回る
```

`repl.py` の `run()` ループは変更不要。`chat()` の中身だけ差し替える。

### → Phase 4: ファイルコンテキスト

`ConversationState` にファイル情報を追加:

```python
@dataclass
class ConversationState:
    messages: list[BaseMessage] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)  # ← 追加
```

### → Phase 6: MCP連携

`_build_llm()` のパターンを拡張して、MCPツールサーバーからツールを動的に読み込む:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

async with MultiServerMCPClient(servers) as client:
    tools = client.get_tools()
    llm_with_tools = self.llm.bind_tools(tools)
```

---

## まとめ: Phase 1で学んだ技術要素チェックリスト

- [ ] REPLパターン（Read-Eval-Print-Loop）
- [ ] LangChainのメッセージモデル（System / Human / AI）
- [ ] BaseChatModelによるプロバイダー抽象化
- [ ] `.stream()` によるトークン単位のストリーミング
- [ ] Rich LiveによるリアルタイムMarkdownレンダリング
- [ ] Command + Registryパターンによる拡張可能なコマンドシステム
- [ ] `TYPE_CHECKING` による循環import回避
- [ ] `@dataclass` + `field(default_factory=...)` の活用
- [ ] `prompt_toolkit` によるTab補完・入力履歴
- [ ] エラー時の状態ロールバック
