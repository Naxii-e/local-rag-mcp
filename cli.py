"""CLI for indexing TypeScript/Go codebases into LanceDB.

Usage:
    python cli.py index /path/to/project     # Full index
    python cli.py update /path/to/project    # Incremental (changed files only)
    python cli.py clear                      # Remove index
    python cli.py status                     # Show index status
"""

import os
import sys
import time
from pathlib import Path

import click
from tqdm import tqdm

import config
from indexer.chunker import Chunker
from indexer.embedder import Embedder
from indexer.storage import LanceDBStorage


@click.group()
def cli():
    """RAG indexer."""
    pass


@cli.command()
@click.argument("path")
def index(path: str):
    """Full index of a directory."""
    target = Path(path)
    if not target.exists() or not target.is_dir():
        click.echo(f"Directory not found: {path}", err=True)
        sys.exit(1)

    start_time = time.time()

    try:
        storage = LanceDBStorage()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    embedder = Embedder()
    chunker = Chunker()

    click.echo(f"Chunking: {path}")
    all_chunks = chunker.chunk_directory(str(target))

    if not all_chunks:
        click.echo("No target files found.")
        return

    file_set = {c["metadata"]["filepath"] for c in all_chunks}
    file_count = len(file_set)
    chunk_count = len(all_chunks)
    click.echo(f"Files: {file_count}, Chunks: {chunk_count}")

    try:
        storage.ensure_collection(vector_size=config.EMBED_DIM)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    batch_size = config.EMBED_BATCH_SIZE
    total_upserted = 0

    with tqdm(total=chunk_count, desc="Indexing", unit="chunk") as pbar:
        for i in range(0, chunk_count, batch_size):
            batch = all_chunks[i : i + batch_size]
            try:
                embeddings = embedder.embed_documents([c["text"] for c in batch])
                upserted = storage.upsert_chunks(
                    [{**chunk, "embedding": emb} for chunk, emb in zip(batch, embeddings)]
                )
                total_upserted += upserted
            except Exception as e:
                click.echo(f"\nError: {e}", err=True)
                sys.exit(1)
            pbar.update(len(batch))

    elapsed = time.time() - start_time
    click.echo(
        f"\nDone: files={file_count}, chunks={chunk_count}, "
        f"upserted={total_upserted}, time={elapsed:.1f}s"
    )


@cli.command()
@click.argument("path")
def update(path: str):
    """Incremental update (re-index changed files only)."""
    target = Path(path)
    if not target.exists() or not target.is_dir():
        click.echo(f"Directory not found: {path}", err=True)
        sys.exit(1)

    start_time = time.time()

    try:
        storage = LanceDBStorage()
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    embedder = Embedder()
    chunker = Chunker()

    target_exts = set(config.TARGET_EXTENSIONS.keys())
    excluded_dirs = config.EXCLUDED_DIRS
    target_files: list[str] = []

    for root, dirs, files in os.walk(str(target)):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for filename in files:
            _, ext = os.path.splitext(filename)
            if ext in target_exts:
                target_files.append(os.path.join(root, filename))

    if not target_files:
        click.echo("No target files found.")
        return

    click.echo(f"Target files: {len(target_files)}")

    try:
        storage.ensure_collection(vector_size=config.EMBED_DIM)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    total_upserted = 0
    total_chunks = 0
    processed_files = 0

    with tqdm(total=len(target_files), desc="Updating", unit="file") as pbar:
        for filepath in target_files:
            # normalize path separators
            normalized_path = filepath.replace("\\", "/")
            existing = storage.get_by_filepath(normalized_path)
            existing_hashes = {c.get("text_hash", "") for c in existing if c.get("text_hash")}

            chunks = chunker.chunk_file(filepath)
            if not chunks:
                pbar.update(1)
                continue

            new_chunks = [
                c for c in chunks
                if c["metadata"]["text_hash"] not in existing_hashes
            ]

            if not new_chunks:
                # no changes
                pbar.update(1)
                continue

            # delete old chunks and re-index
            if existing:
                try:
                    storage.delete_by_filepath(normalized_path)
                except Exception as e:
                    click.echo(f"\nError ({filepath}): {e}", err=True)
                    pbar.update(1)
                    continue

            batch_size = config.EMBED_BATCH_SIZE
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                try:
                    embeddings = embedder.embed_documents([c["text"] for c in batch])
                    upserted = storage.upsert_chunks(
                        [{**chunk, "embedding": emb} for chunk, emb in zip(batch, embeddings)]
                    )
                    total_upserted += upserted
                    total_chunks += len(batch)
                except Exception as e:
                    click.echo(f"\nError ({filepath}): {e}", err=True)
                    sys.exit(1)

            processed_files += 1
            pbar.update(1)

    elapsed = time.time() - start_time
    click.echo(
        f"\nDone: files_updated={processed_files}, "
        f"chunks={total_chunks}, upserted={total_upserted}, "
        f"time={elapsed:.1f}s"
    )


@cli.command()
def clear():
    """Remove the index."""
    try:
        storage = LanceDBStorage()
        storage.clear_collection()
        click.echo("Index cleared.")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """Show index status."""
    try:
        storage = LanceDBStorage()
        stats = storage.get_stats()
        click.echo(f"Collection: {stats['collection_name']}")
        click.echo(f"Total chunks: {stats['total_chunks']}")
        click.echo(f"DB path: {stats['db_path']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
