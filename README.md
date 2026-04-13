# local-rag-mcp

[日本語](README.ja.md)

## Overview

A local semantic search system for TypeScript and Go codebases, accessible from Claude Code via MCP.

Source files are chunked using tree-sitter AST parsing, embedded with `intfloat/multilingual-e5-large`, stored in LanceDB, and reranked with `BAAI/bge-reranker-v2-m3`.

## Requirements

- Python 3.10+
- NVIDIA GPU (3-4 GB VRAM)

### Tested on

| GPU | OS | CUDA | Python |
|---|---|---|---|
| RTX 5060 Ti | Windows 11 Pro 25H2 | 13.2 | 3.10 |
| RTX 5070 Ti | Windows 11 Pro 25H2 | 13.2 | 3.10 |

## Installation

Install PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install remaining dependencies:

```bash
pip install -r requirements.txt
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Indexing

```bash
python cli.py index /path/to/project    # full index
python cli.py update /path/to/project   # incremental (changed files only)
python cli.py status                    # show chunk count and DB path
python cli.py clear                     # drop the index
```

> [!NOTE]
> Supported file types: `.ts`, `.tsx`, `.go`.

## MCP Server

Register with Claude Code:

```bash
claude mcp add --scope global codebase-rag \
  python /path/to/local-rag-mcp/rag_server/server.py
```

| Tool | Description |
|---|---|
| `search_codebase(query: str, n_results: int = 8, language: str \| None = None)` | semantic search using embedding and rerank |
| `search_by_filepath(pattern: str, n_results: int = 20)` | search by filepath substring |
| `get_file_summary(filepath: str)` | retrieve all indexed chunks for a file |

## Visualizer

```bash
python visualizer/app.py
```

Opens at `http://localhost:8765`. The port can be changed via the `PORT` environment variable.

### Stats

Total chunk and file counts, broken down by language and node type.

### Files

Per-file chunk count, sorted by volume.

### Scatter

2D projection of all embedding vectors using UMAP, falling back to PCA if unavailable. Colored by language and node type. Results are cached in `lancedb_data/umap_cache.json`.

### Search

Semantic search with embedding and rerank, returning ranked results with scores and code snippets.

## Auto-reindex Hook

To re-index automatically on every file edit in Claude Code, add the following to `.claude/settings.json`:

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

Set `PROJECT_DIR` in `hooks/post_edit_reindex.py` to the target codebase path before use.

## Configuration

All settings are constants in `config.py`.

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` | Embedding model |
| `EMBED_BATCH_SIZE` | `64` | Batch size for embedding |
| `EMBED_DIM` | `1024` | Vector dimension |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `RERANK_TOP_K` | `40` | Candidates fetched from LanceDB |
| `FINAL_TOP_K` | `8` | Results returned after rerank |
| `CHUNK_MIN_CHARS` | `30` | Minimum chunk size (chars) |
| `CHUNK_MAX_CHARS` | `1500` | Maximum chunk size (chars) |
| `LANCEDB_PATH` | `./lancedb_data` | Vector DB directory |

To override the compute device:

```bash
EMBED_DEVICE=cpu python cli.py index /path/to/project
```
