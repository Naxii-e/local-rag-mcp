# local-rag-mcp

[日本語](README.ja.md)

## Overview

A local semantic search system for TypeScript and Go codebases, accessible from Claude Code via MCP.

Source files are chunked using tree-sitter AST parsing, embedded with `intfloat/multilingual-e5-large`, stored in LanceDB, and reranked with `BAAI/bge-reranker-v2-m3`.

## Requirements

- Python 3.10+
- One of:
  - NVIDIA GPU (3-4 GB VRAM, CUDA)
  - Apple Silicon (Metal Performance Shaders / MPS)
  - CPU (slow fallback)

### Tested on

| Device | OS | Backend | Python |
|---|---|---|---|
| RTX 5060 Ti | Windows 11 Pro 25H2 | CUDA 13.2 | 3.10 |
| RTX 5070 Ti | Windows 11 Pro 25H2 | CUDA 13.2 | 3.10 |

## Installation

(Recommended) Create a virtual environment with [uv](https://docs.astral.sh/uv/):

```bash
uv venv venv
source venv/bin/activate          # macOS / Linux
# .\venv\Scripts\activate         # Windows PowerShell
```

> [!NOTE]
> `uv venv venv` creates the venv at `./venv` to match the path used in the examples below. The default `uv venv` would create `./.venv` instead — adjust the paths accordingly if you prefer that layout.

Install PyTorch.

For NVIDIA GPU (CUDA):

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For Apple Silicon (MPS) or CPU-only:

```bash
uv pip install torch torchvision torchaudio
```

Then install remaining dependencies:

```bash
uv pip install -r requirements.txt
```

> [!TIP]
> Plain `pip install ...` works too if you don't use uv. Replace `uv pip` with `pip` in the commands above.

Verify accelerator availability:

```bash
# CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Apple Silicon
python -c "import torch; print(torch.backends.mps.is_available())"
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

Register with Claude Code. Point `claude mcp add` at the **venv's** Python interpreter so the MCP server picks up the dependencies installed in that environment:

macOS / Linux:

```bash
claude mcp add --scope user codebase-rag \
  /path/to/local-rag-mcp/venv/bin/python \
  /path/to/local-rag-mcp/rag_server/server.py
```

Windows (PowerShell):

```powershell
claude mcp add --scope user codebase-rag `
  C:\path\to\local-rag-mcp\venv\Scripts\python.exe `
  C:\path\to\local-rag-mcp\rag_server\server.py
```

> [!IMPORTANT]
> Always pass the **absolute path** to the venv's Python (`venv/bin/python` or `venv\Scripts\python.exe`). The bare `python` command in your shell PATH is typically not the venv interpreter when Claude Code launches the MCP server, which results in `ModuleNotFoundError` for `lancedb`, `sentence-transformers`, etc.

Alternatively, use `uv run` so uv resolves the project's environment for you:

```bash
claude mcp add --scope user codebase-rag \
  uv --directory /path/to/local-rag-mcp run python rag_server/server.py
```

Verify the registration:

```bash
claude mcp list
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

Device selection is automatic in this order: `EMBED_DEVICE` env var → CUDA → Apple MPS → CPU. The reranker honors `RERANK_DEVICE` first, then falls back to `EMBED_DEVICE`.

To override the compute device:

```bash
# Force CPU
EMBED_DEVICE=cpu python cli.py index /path/to/project

# Explicit Apple Silicon
EMBED_DEVICE=mps python cli.py index /path/to/project
```
