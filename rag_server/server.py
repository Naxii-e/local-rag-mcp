"""MCP server (stdio transport) providing codebase search tools."""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

# add project root to sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from indexer.embedder import Embedder
from indexer.reranker import Reranker
from indexer.storage import LanceDBStorage
import config

logger = logging.getLogger(__name__)

mcp = FastMCP("codebase-rag")

# lazy init on first tool call
_embedder: Embedder | None = None
_reranker: Reranker | None = None
_storage: LanceDBStorage | None = None


def get_storage() -> LanceDBStorage:
    """Lazily initialize and return storage (no embedder/reranker needed)."""
    global _storage
    if _storage is None:
        _storage = LanceDBStorage()
    return _storage


def get_components() -> tuple[Embedder, Reranker, LanceDBStorage]:
    """Lazily initialize and return embedder, reranker, and storage."""
    global _embedder, _reranker
    if _embedder is None:
        _embedder = Embedder()
    if _reranker is None:
        _reranker = Reranker()
    return _embedder, _reranker, get_storage()


# Tools

@mcp.tool()
def search_codebase(query: str, n_results: int = 8, language: str | None = None) -> str:
    """
    Semantic search over the indexed codebase (embedding + rerank).

    Args:
        query: Natural language or keyword query.
        n_results: Number of results to return (default: 8).
        language: Filter by language ("typescript" or "go"). None = all.

    Returns:
        Markdown with file paths, line numbers, code snippets, and scores.
    """
    try:
        embedder, reranker, storage = get_components()

        query_vector = embedder.embed_query(query)
        candidates = storage.search(query_vector, top_k=config.RERANK_TOP_K, language=language)
        results = reranker.rerank(query, candidates, top_k=n_results)

        return _format_results(results)
    except ConnectionError as exc:
        logger.error("LanceDB connection error: %s", exc)
        return f"LanceDB connection failed: {exc}"
    except RuntimeError as exc:
        logger.error("Component init error: %s", exc, exc_info=True)
        return f"Component initialization failed (model not downloaded or insufficient VRAM): {exc}"
    except Exception as exc:
        logger.error("search_codebase error: %s", exc, exc_info=True)
        return f"Unexpected error: {exc}"


@mcp.tool()
def search_by_filepath(pattern: str, n_results: int = 20) -> str:
    """
    Search chunks by filepath keyword (e.g. "auth", "handler").

    Args:
        pattern: Substring to match against file paths.
        n_results: Number of results to return (default: 20).

    Returns:
        Markdown listing matched chunks.
    """
    try:
        storage = get_storage()
        results = storage.search_by_filepath(pattern, limit=n_results)
        return _format_results(results)
    except ConnectionError as exc:
        logger.error("LanceDB connection error: %s", exc)
        return f"LanceDB connection failed: {exc}"
    except RuntimeError as exc:
        logger.error("Component init error: %s", exc, exc_info=True)
        return f"Component initialization failed: {exc}"
    except Exception as exc:
        logger.error("search_by_filepath error: %s", exc, exc_info=True)
        return f"Unexpected error: {exc}"


@mcp.tool()
def get_file_summary(filepath: str) -> str:
    """
    Retrieve all indexed chunks for a specific file.

    Args:
        filepath: Relative file path (e.g. "src/auth/handler.ts").

    Returns:
        Markdown listing all chunks for that file.
    """
    try:
        storage = get_storage()
        results = storage.get_by_filepath(filepath)
        if not results:
            return f"No chunks found for '{filepath}'. The file may not be indexed."
        return _format_results(results)
    except ConnectionError as exc:
        logger.error("LanceDB connection error: %s", exc)
        return f"LanceDB connection failed: {exc}"
    except RuntimeError as exc:
        logger.error("Component init error: %s", exc, exc_info=True)
        return f"Component initialization failed: {exc}"
    except Exception as exc:
        logger.error("get_file_summary error: %s", exc, exc_info=True)
        return f"Unexpected error: {exc}"


# Formatting

def _format_results(results: list[dict[str, Any]]) -> str:
    """Format search results as Markdown."""
    if not results:
        return "No results found."

    lines = []
    for i, r in enumerate(results, 1):
        filepath = r.get("filepath", "unknown")
        start_line = r.get("start_line", 0)
        end_line = r.get("end_line", 0)
        node_type = r.get("node_type", "")
        language = r.get("language", "")
        score = r.get("rerank_score") if "rerank_score" in r else r.get("score")
        text = r.get("text", "")

        lines.append(f"## {i}. {filepath}:{start_line}-{end_line} ({node_type})")
        if score is not None:
            lines.append(f"Score: {score:.4f}")
        if language:
            lines.append(f"Language: {language}")
        lines.append(f"```{language}")
        lines.append(text)
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


# Entry point

if __name__ == "__main__":
    mcp.run(transport="stdio")
