# local-rag-mcp

[English](README.md)

## 概要

TypeScript/GoコードベースのセマンティックRAG検索システムです。Claude Code からMCP経由で利用できます。

tree-sitter によるAST解析でチャンク分割し、`intfloat/multilingual-e5-large` でEmbedding、LanceDB にベクトルを保存し、`BAAI/bge-reranker-v2-m3` でRerankします。

## 要件

- Python 3.10+
- NVIDIA GPU（3〜4 GB VRAM）

### 動作確認済み環境

| GPU | OS | CUDA | Python |
|---|---|---|---|
| RTX 5060 Ti | Windows 11 Pro 25H2 | 13.2 | 3.10 |
| RTX 5070 Ti | Windows 11 Pro 25H2 | 13.2 | 3.10 |

## インストール

PyTorch をインストールします。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

残りの依存パッケージをインストールします。

```bash
pip install -r requirements.txt
```

GPUが認識されているか確認します。

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
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

Claude Code に登録します。

```bash
claude mcp add --scope global codebase-rag \
  python /path/to/local-rag-mcp/rag_server/server.py
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

デバイスを強制指定するには環境変数を設定します。

```bash
EMBED_DEVICE=cpu python cli.py index /path/to/project
```
