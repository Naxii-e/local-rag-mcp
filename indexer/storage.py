"""LanceDB storage layer for chunk upsert, search, and deletion."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import lancedb
import pyarrow as pa

import config

logger = logging.getLogger(__name__)


def _escape_sql_str(value: str) -> str:
    """Escape single quotes for LanceDB SQL strings."""
    return value.replace("'", "''")


def _where_contains(table, column: str, value: str, limit: int) -> list:
    """Search with contains(), falling back to LIKE."""
    safe = _escape_sql_str(value)
    try:
        return (
            table.search()
            .where(f"contains({column}, '{safe}')", prefilter=True)
            .limit(limit)
            .to_list()
        )
    except Exception:
        return (
            table.search()
            .where(f"{column} LIKE '%{safe}%'", prefilter=True)
            .limit(limit)
            .to_list()
        )


class LanceDBStorage:
    """LanceDB-backed chunk storage."""

    def __init__(self) -> None:
        try:
            self.db = lancedb.connect(config.LANCEDB_PATH)
            self.table_name = config.LANCEDB_TABLE
            self._table = None  # lazy init
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to LanceDB ({config.LANCEDB_PATH}): {exc}"
            ) from exc

    # Table management

    def ensure_collection(self, vector_size: int) -> None:
        """Create the table if absent and build an HNSW index."""
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_size)),
            pa.field("text", pa.string()),
            pa.field("filepath", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("node_type", pa.string()),
            pa.field("language", pa.string()),
            pa.field("text_hash", pa.string()),
        ])

        if self.table_name not in self.db.table_names():
            self._table = self.db.create_table(self.table_name, schema=schema)
        else:
            self._table = self.db.open_table(self.table_name)

        # build HNSW index; skip if already exists
        try:
            self._table.create_index(
                "vector",
                config=lancedb.index.IvfHnswSq(
                    distance_type="cosine",
                ),
            )
        except Exception as exc:
            logger.debug("HNSW index skipped (exists or unsupported): %s", exc)
            # fallback for older LanceDB API
            try:
                self._table.create_index(metric="cosine", index_type="IVF_HNSW_SQ")
            except Exception as exc2:
                logger.debug("Fallback index also skipped: %s", exc2)
                # brute-force search still works without an index

    # Internal helpers

    def _get_table(self):
        if self._table is None:
            if self.table_name in self.db.table_names():
                self._table = self.db.open_table(self.table_name)
            else:
                raise RuntimeError(
                    "Table not initialized. Call ensure_collection() first."
                )
        return self._table

    # Upsert

    def upsert_chunks(self, chunks: list[dict]) -> int:
        """Upsert chunks; skip duplicates by text_hash. Returns inserted count."""
        if not chunks:
            return 0

        table = self._get_table()

        # fetch existing hashes with column projection to avoid full load
        try:
            existing_hashes = set(
                r["text_hash"]
                for r in table.search()
                .select(["text_hash"])
                .limit(1_000_000)
                .to_list()
                if r.get("text_hash")
            )
        except Exception:
            existing_hashes = set()

        rows = []
        seen_in_batch: set[str] = set()

        for chunk in chunks:
            text_hash = chunk["metadata"]["text_hash"]
            if text_hash in existing_hashes or text_hash in seen_in_batch:
                continue
            seen_in_batch.add(text_hash)

            rows.append({
                "id": str(uuid.uuid4()),
                "vector": chunk["embedding"],
                "text": chunk["text"],
                "filepath": chunk["metadata"].get("filepath", ""),
                "start_line": chunk["metadata"].get("start_line", 0),
                "end_line": chunk["metadata"].get("end_line", 0),
                "node_type": chunk["metadata"].get("node_type", ""),
                "language": chunk["metadata"].get("language", ""),
                "text_hash": text_hash,
            })

        if rows:
            BATCH_SIZE = 128
            for i in range(0, len(rows), BATCH_SIZE):
                table.add(rows[i:i + BATCH_SIZE])

        return len(rows)

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 40,
        language: str | None = None,
    ) -> list[dict]:
        """Vector search. Optionally filter by language."""
        table = self._get_table()

        query = table.search(query_vector, vector_column_name="vector").limit(top_k)

        if language:
            safe_language = _escape_sql_str(language)
            query = query.where(f"language = '{safe_language}'", prefilter=True)

        results = query.to_list()

        return [
            {
                "text": row["text"],
                "score": 1.0 - float(row["_distance"]),  # cosine similarity
                "filepath": row["filepath"],
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "node_type": row["node_type"],
                "language": row["language"],
                "text_hash": row["text_hash"],
            }
            for row in results
        ]

    def search_by_filepath(self, pattern: str, limit: int = 20) -> list[dict]:
        """Search chunks by filepath substring."""
        table = self._get_table()
        results = _where_contains(table, "filepath", pattern, limit)
        return [
            {
                "text": row["text"],
                "filepath": row["filepath"],
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "node_type": row["node_type"],
                "language": row["language"],
                "text_hash": row["text_hash"],
            }
            for row in results
        ]

    # File-level operations

    def get_by_filepath(self, filepath: str) -> list[dict]:
        """Return all chunks for a given filepath (partial match)."""
        table = self._get_table()
        normalized = filepath.replace("\\", "/")
        results = _where_contains(table, "filepath", normalized, 500)
        return [
            {
                "text": row["text"],
                "filepath": row["filepath"],
                "start_line": int(row["start_line"]),
                "end_line": int(row["end_line"]),
                "node_type": row["node_type"],
                "language": row["language"],
                "text_hash": row["text_hash"],
            }
            for row in results
        ]

    def delete_by_filepath(self, filepath: str) -> int:
        """Delete all chunks for a filepath (exact match). Returns deleted count."""
        table = self._get_table()
        safe_filepath = _escape_sql_str(filepath)
        # count before delete
        results = (
            table.search()
            .where(f"filepath = '{safe_filepath}'", prefilter=True)
            .limit(10_000)
            .to_list()
        )
        count = len(results)
        if count > 0:
            table.delete(f"filepath = '{safe_filepath}'")
        return count

    # Stats / management

    def get_stats(self) -> dict:
        """Return table statistics: collection_name, total_chunks, db_path."""
        try:
            table = self._get_table()
            try:
                total = table.count_rows()
            except AttributeError:
                # fallback for older LanceDB without count_rows()
                total = len(table.to_list())
            return {
                "collection_name": self.table_name,
                "total_chunks": total,
                "db_path": config.LANCEDB_PATH,
            }
        except RuntimeError:
            return {
                "collection_name": self.table_name,
                "total_chunks": 0,
                "db_path": config.LANCEDB_PATH,
            }

    def clear_collection(self) -> None:
        """Drop the table."""
        if self.table_name in self.db.table_names():
            self.db.drop_table(self.table_name)
        self._table = None
