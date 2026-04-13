"""Tests for visualizer/app.py endpoints."""
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

def _make_mock_table(rows):
    table = MagicMock()
    (
        table.search.return_value
        .select.return_value
        .limit.return_value
        .to_list.return_value
    ) = rows
    table.count_rows.return_value = len(rows)
    return table

SAMPLE_ROWS = [
    {"language": "go",         "filepath": "src/main.go",    "node_type": "function_declaration",
     "vector": [0.1] * 1024,   "text": "func main() {}",     "start_line": 1, "end_line": 5,
     "text_hash": "abc"},
    {"language": "typescript", "filepath": "src/app.ts",     "node_type": "class_declaration",
     "vector": [0.2] * 1024,   "text": "class App {}",       "start_line": 1, "end_line": 10,
     "text_hash": "def"},
]

@pytest.fixture
def client():
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from visualizer.app import app
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_storage(monkeypatch):
    import visualizer.app as vapp
    storage = MagicMock()
    storage._get_table.return_value = _make_mock_table(SAMPLE_ROWS)
    monkeypatch.setattr(vapp, "_storage", storage)
    return storage


def test_stats_returns_counts(client):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_chunks"] == 2
    assert body["total_files"] == 2
    assert body["language_breakdown"]["go"] == 1
    assert body["language_breakdown"]["typescript"] == 1


def test_files_returns_sorted_list(client):
    resp = client.get("/api/files")
    assert resp.status_code == 200
    files = resp.json()
    assert len(files) == 2
    assert "filepath" in files[0]
    assert "chunk_count" in files[0]
    assert "language" in files[0]


def test_umap_status_idle_when_no_cache(client, tmp_path, monkeypatch):
    import visualizer.app as vapp
    monkeypatch.setattr(vapp, "UMAP_CACHE_PATH", tmp_path / "umap_cache.json")
    monkeypatch.setattr(vapp, "_umap_state", {"status": "idle", "message": ""})
    resp = client.get("/api/umap/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "idle"

def test_umap_404_when_no_cache(client, tmp_path, monkeypatch):
    import visualizer.app as vapp
    monkeypatch.setattr(vapp, "UMAP_CACHE_PATH", tmp_path / "umap_cache.json")
    resp = client.get("/api/umap")
    assert resp.status_code == 404

def test_umap_returns_cached_data(client, tmp_path, monkeypatch):
    import json, visualizer.app as vapp
    cache = tmp_path / "umap_cache.json"
    payload = {"total": 2, "method": "PCA", "data": [{"x": 0.1, "y": 0.2}]}
    cache.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(vapp, "UMAP_CACHE_PATH", cache)
    resp = client.get("/api/umap")
    assert resp.status_code == 200
    assert resp.json()["method"] == "PCA"


def test_search_returns_results(client, monkeypatch):
    import visualizer.app as vapp

    fake_embedder = MagicMock()
    fake_embedder.embed_query.return_value = [0.1] * 1024
    monkeypatch.setattr(vapp, "_embedder", fake_embedder)

    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [
        {"text": "func main(){}", "filepath": "src/main.go",
         "start_line": 1, "end_line": 5, "node_type": "function_declaration",
         "language": "go", "rerank_score": 0.95}
    ]
    monkeypatch.setattr(vapp, "_reranker", fake_reranker)

    resp = client.get("/api/search?q=main+function")
    assert resp.status_code == 200
    results = resp.json()
    assert len(results) == 1
    assert results[0]["rerank_score"] == 0.95
