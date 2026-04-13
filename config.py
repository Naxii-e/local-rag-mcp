"""Configuration for local RAG + MCP server."""

import os


# LanceDB

LANCEDB_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lancedb_data")
LANCEDB_TABLE: str = "codebase"

# Embedding

EMBED_MODEL: str = "intfloat/multilingual-e5-large"
EMBED_BATCH_SIZE: int = 64
EMBED_DIM: int = 1024  # e5-large output dim

# Reranker

RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K: int = 40   # initial candidates fetched from LanceDB
FINAL_TOP_K: int = 8     # final results after rerank
RERANKER_MAX_LENGTH: int = 1024

if FINAL_TOP_K > RERANK_TOP_K:
    raise ValueError(f"FINAL_TOP_K ({FINAL_TOP_K}) must be <= RERANK_TOP_K ({RERANK_TOP_K})")

# Chunker

CHUNK_MIN_CHARS: int = 30
CHUNK_MAX_CHARS: int = 1500
EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".claude",
    ".worktrees",
    "node_modules",
    "dist",
    "build",
    ".next",
    "vendor",
    "__pycache__",
    "coverage",
    ".turbo",
})
TARGET_EXTENSIONS: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
}
