"""RAG index visualizer backend.

Usage: python visualizer/app.py → http://localhost:8765
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config
from indexer.storage import LanceDBStorage

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Index Visualizer")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_storage: LanceDBStorage | None = None
_embedder = None
_reranker = None
_umap_state: dict = {"status": "idle", "message": ""}
_executor = ThreadPoolExecutor(max_workers=1)

UMAP_CACHE_PATH = Path(config.LANCEDB_PATH) / "umap_cache.json"


def _get_storage() -> LanceDBStorage:
    global _storage
    if _storage is None:
        _storage = LanceDBStorage()
    return _storage


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(
        (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
    )


@app.get("/api/stats")
def get_stats():
    table = _get_storage()._get_table()
    rows = (
        table.search()
        .select(["language", "filepath", "node_type"])
        .limit(1_000_000)
        .to_list()
    )
    lang_count: dict[str, int] = {}
    node_count: dict[str, int] = {}
    for r in rows:
        lang = r.get("language") or "unknown"
        lang_count[lang] = lang_count.get(lang, 0) + 1
        node = r.get("node_type") or "unknown"
        node_count[node] = node_count.get(node, 0) + 1
    return {
        "total_chunks": len(rows),
        "total_files": len({r["filepath"] for r in rows}),
        "language_breakdown": lang_count,
        "node_type_breakdown": node_count,
    }


@app.get("/api/files")
def get_files():
    table = _get_storage()._get_table()
    rows = (
        table.search()
        .select(["filepath", "language"])
        .limit(1_000_000)
        .to_list()
    )
    file_stats: dict[str, dict] = {}
    for r in rows:
        fp = r["filepath"]
        if fp not in file_stats:
            file_stats[fp] = {
                "filepath": fp,
                "language": r.get("language") or "",
                "chunk_count": 0,
            }
        file_stats[fp]["chunk_count"] += 1
    return sorted(file_stats.values(), key=lambda x: x["chunk_count"], reverse=True)


def _compute_umap() -> None:
    global _umap_state
    try:
        _umap_state = {"status": "computing", "message": "Loading vectors..."}
        table = _get_storage()._get_table()
        rows = (
            table.search()
            .select(["vector", "filepath", "language", "node_type", "text", "start_line"])
            .limit(1_000_000)
            .to_list()
        )
        n = len(rows)
        if n < 10:
            _umap_state = {"status": "error", "message": "Insufficient data (min 10)"}
            return

        vectors = np.array([r["vector"] for r in rows], dtype=np.float32)
        _umap_state = {"status": "computing", "message": f"Reducing {n:,} to 2D..."}

        try:
            import umap as umap_lib
            embedding = umap_lib.UMAP(
                n_components=2, random_state=42, n_neighbors=15, min_dist=0.1
            ).fit_transform(vectors)
            method = "UMAP"
        except ImportError:
            from sklearn.decomposition import PCA
            embedding = PCA(n_components=2).fit_transform(vectors)
            method = "PCA"

        data = []
        for i, r in enumerate(rows):
            fp = r.get("filepath") or ""
            parts = fp.replace("\\", "/").split("/")
            short_fp = "/".join(parts[-3:]) if len(parts) >= 3 else fp
            snippet = (r.get("text") or "").replace("\n", " ")[:120]
            data.append({
                "x": float(embedding[i, 0]),
                "y": float(embedding[i, 1]),
                "filepath": fp,
                "short_filepath": short_fp,
                "language": r.get("language") or "",
                "node_type": r.get("node_type") or "",
                "start_line": int(r.get("start_line") or 0),
                "snippet": snippet,
            })

        UMAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        UMAP_CACHE_PATH.write_text(
            json.dumps({"total": n, "method": method, "data": data}, ensure_ascii=False),
            encoding="utf-8",
        )
        _umap_state = {"status": "ready", "message": f"{method} done ({n:,})"}
    except Exception as exc:
        _umap_state = {"status": "error", "message": str(exc)}
        logger.exception("UMAP computation error")


@app.get("/api/umap/status")
def umap_status():
    if _umap_state["status"] == "idle" and UMAP_CACHE_PATH.exists():
        return {"status": "ready", "message": "Cache available"}
    return _umap_state


@app.post("/api/umap/compute")
async def compute_umap():
    global _umap_state
    if _umap_state["status"] == "computing":
        return {"status": "already_computing"}
    _umap_state = {"status": "computing", "message": "Starting..."}
    asyncio.get_event_loop().run_in_executor(_executor, _compute_umap)
    return {"status": "started"}


@app.delete("/api/umap/cache")
def clear_umap_cache():
    global _umap_state
    if UMAP_CACHE_PATH.exists():
        UMAP_CACHE_PATH.unlink()
    _umap_state = {"status": "idle", "message": ""}
    return {"status": "cleared"}


@app.get("/api/umap")
def get_umap():
    if not UMAP_CACHE_PATH.exists():
        raise HTTPException(status_code=404, detail="Click Generate first")
    try:
        return json.loads(UMAP_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _get_embedder():
    global _embedder
    if _embedder is None:
        from indexer.embedder import Embedder
        _embedder = Embedder()
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        from indexer.reranker import Reranker
        _reranker = Reranker()
    return _reranker


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    n: int = Query(8, ge=1, le=50),
    language: str | None = None,
):
    embedder = _get_embedder()
    storage = _get_storage()
    reranker = _get_reranker()
    qv = embedder.embed_query(q)
    candidates = storage.search(qv, top_k=config.RERANK_TOP_K, language=language or None)
    return reranker.rerank(q, candidates, top_k=n)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8765))
    print(f"\n  RAG Index Visualizer → http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
