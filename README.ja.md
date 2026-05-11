# local-rag-mcp

[English](README.md)

## 概要

TypeScript/GoコードベースのセマンティックRAG検索システムです。Claude Code からMCP経由で利用できます。

tree-sitter によるAST解析でチャンク分割し、`intfloat/multilingual-e5-large` でEmbedding、LanceDB にベクトルを保存し、`BAAI/bge-reranker-v2-m3` でRerankします。

## 要件

- Python 3.10+
- 次のいずれか:
  - NVIDIA GPU（3〜4 GB VRAM、CUDA）
  - Apple Silicon（Metal Performance Shaders / MPS）
  - CPU（低速フォールバック）

### 動作確認済み環境

| デバイス | OS | バックエンド | Python |
|---|---|---|---|
| RTX 5060 Ti | Windows 11 Pro 25H2 | CUDA 13.2 | 3.10 |
| RTX 5070 Ti | Windows 11 Pro 25H2 | CUDA 13.2 | 3.10 |

## インストール

（推奨）[uv](https://docs.astral.sh/uv/) で仮想環境を作成します。

```bash
uv venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows PowerShell
```

> [!NOTE]
> 以降のサンプルで使うパスに合わせて、`uv venv venv` で `./venv` に作成しています。デフォルトの `uv venv` は `./.venv` に作成されるので、その場合は以降のパスを読み替えてください。

PyTorch をインストールします。

NVIDIA GPU（CUDA）の場合:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Apple Silicon（MPS）または CPU のみの場合:

```bash
uv pip install torch torchvision torchaudio
```

残りの依存パッケージをインストールします。

```bash
uv pip install -r requirements.txt
```

> [!TIP]
> uv を使わない場合は、上記の `uv pip` を `pip` に置き換えても構いません。

アクセラレーターが認識されているか確認します。

```bash
# CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
```

## インデックス作成

```bash
python cli.py index /path/to/project    # フルインデックス
python cli.py update /path/to/project   # 差分更新（変更ファイルのみ）
python cli.py status                    # チャンク数・DBパス表示
python cli.py clear                     # インデックス削除
```

> [!NOTE]
> 対応ファイルは `.ts`、`.tsx`、`.go` です。

## MCPサーバー

Claude Code に登録します。`claude mcp add` には **venv の Python** の絶対パスを指定してください。これにより、venv にインストールした依存パッケージが MCP サーバーから参照されます。

macOS / Linux:

```bash
claude mcp add --scope user codebase-rag \
  /path/to/local-rag-mcp/venv/bin/python \
  /path/to/local-rag-mcp/rag_server/server.py
```

Windows（PowerShell）:

```powershell
claude mcp add --scope user codebase-rag `
  C:\path\to\local-rag-mcp\venv\Scripts\python.exe `
  C:\path\to\local-rag-mcp\rag_server\server.py
```

> [!IMPORTANT]
> 必ず venv の Python（`venv/bin/python` または `venv\Scripts\python.exe`）の**絶対パス**を指定してください。シェル上の `python` は Claude Code が MCP サーバーを起動するときには venv の Python とは限らず、その場合 `lancedb` や `sentence-transformers` 等の `ModuleNotFoundError` になります。

登録結果を確認します。

```bash
claude mcp list
```

| ツール | 説明 |
|---|---|
| `search_codebase(query: str, n_results: int = 8, language: str \| None = None)` | Embedding + Rerank によるセマンティック検索 |
| `search_by_filepath(pattern: str, n_results: int = 20)` | ファイルパスのキーワード検索 |
| `get_file_summary(filepath: str)` | 特定ファイルの全チャンク取得 |

## ビジュアライザー

```bash
python visualizer/app.py
```

`http://localhost:8765` で起動します。ポートは `PORT` 環境変数で変更できます。

### Stats

総チャンク数・ファイル数を言語別・ノードタイプ別に表示します。

### Files

ファイル別チャンク数の一覧をチャンク数の多い順に表示します。

### Scatter

全Embeddingベクトルを UMAPで2次元に次元削減した散布図です。UMAPが利用できない場合はPCAにフォールバックします。言語・ノードタイプで色分けされ、結果は `lancedb_data/umap_cache.json` にキャッシュされます。

### Search

Embedding + Rerank によるセマンティック検索です。スコアとコードスニペット付きで結果を返します。

## 自動再インデックスフック

Claude Code でのファイル編集後に自動再インデックスするには、`.claude/settings.json` に以下を追加してください。

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit",
        "hooks": [
          {
            "type": "command",
            "command": "python /path/to/local-rag-mcp/hooks/post_edit_reindex.py"
          }
        ]
      }
    ]
  }
}
```

使用前に `hooks/post_edit_reindex.py` の `PROJECT_DIR` を対象コードベースのパスに変更してください。

## 設定

設定はすべて `config.py` の定数です。

| 定数 | デフォルト | 説明 |
|---|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | Embeddingモデル |
| `EMBED_BATCH_SIZE` | `64` | Embeddingバッチサイズ |
| `EMBED_DIM` | `1024` | ベクトル次元数 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Rerankerモデル |
| `RERANK_TOP_K` | `40` | LanceDB初回取得件数 |
| `FINAL_TOP_K` | `8` | Rerank後の返却件数 |
| `CHUNK_MIN_CHARS` | `30` | チャンク最小文字数 |
| `CHUNK_MAX_CHARS` | `1500` | チャンク最大文字数 |
| `LANCEDB_PATH` | `./lancedb_data` | ベクトルDBの保存先 |

デバイスは自動選択されます。優先順位は `EMBED_DEVICE` 環境変数 → CUDA → Apple MPS → CPU です。Reranker は `RERANK_DEVICE` を先に参照し、未設定なら `EMBED_DEVICE` にフォールバックします。

デバイスを強制指定するには環境変数を設定します。

```bash
# CPUを強制
EMBED_DEVICE=cpu python cli.py index /path/to/project

# Apple Silicon を明示
EMBED_DEVICE=mps python cli.py index /path/to/project
```
