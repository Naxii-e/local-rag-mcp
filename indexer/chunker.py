"""AST-based chunker for TypeScript and Go using tree-sitter."""

import hashlib
import os
import sys

# add project root to sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

from tree_sitter import Language, Parser
import tree_sitter_typescript as tsts
import tree_sitter_go as tsg


# Language / parser init

TS_LANGUAGE = Language(tsts.language_typescript())
TSX_LANGUAGE = Language(tsts.language_tsx())
GO_LANGUAGE = Language(tsg.language())

_ts_parser = Parser(TS_LANGUAGE)
_tsx_parser = Parser(TSX_LANGUAGE)
_go_parser = Parser(GO_LANGUAGE)


# Target node types

TS_TARGET_NODES: frozenset[str] = frozenset({
    "function_declaration",
    "method_definition",
    "class_declaration",
    "interface_declaration",
    "type_alias_declaration",
    "export_statement",
    "arrow_function",
})

GO_TARGET_NODES: frozenset[str] = frozenset({
    "function_declaration",
    "method_declaration",
    "type_declaration",
})


# Helpers

def _make_chunk(
    text: str,
    filepath: str,
    start_line: int,
    end_line: int,
    node_type: str,
    language: str,
) -> dict:
    """Build a chunk dict."""
    text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    return {
        "text": text,
        "metadata": {
            "filepath": filepath,
            "start_line": start_line,
            "end_line": end_line,
            "node_type": node_type,
            "language": language,
            "text_hash": text_hash,
        },
    }


def _split_long_chunk(
    code: str,            # raw code without header
    filepath: str,
    start_line: int,
    node_type: str,
    language: str,
    max_chars: int,
) -> list[dict]:
    """Split an oversized node into line-based sub-chunks."""
    lines = code.split("\n")
    chunks: list[dict] = []
    current_lines: list[str] = []
    current_start = start_line

    min_chars = config.CHUNK_MIN_CHARS
    for i, line in enumerate(lines):
        current_lines.append(line)
        if len("\n".join(current_lines)) >= max_chars:
            if len(current_lines) == 1:
                # single oversized line: emit as-is
                chunk_text = current_lines[0]
                end_line = current_start
                if len(chunk_text) >= min_chars:
                    full_text = f"// File: {filepath}\n{chunk_text}"
                    chunks.append(_make_chunk(full_text, filepath, current_start, end_line, node_type, language))
                current_lines = []
                current_start = start_line + i + 1
            else:
                # flush all but the last line
                chunk_text = "\n".join(current_lines[:-1])
                end_line = current_start + len(current_lines) - 2
                if chunk_text and len(chunk_text) >= min_chars:
                    full_text = f"// File: {filepath}\n{chunk_text}"
                    chunks.append(_make_chunk(full_text, filepath, current_start, end_line, node_type, language))
                current_start = start_line + i
                current_lines = [line]

    # flush remaining lines
    if current_lines:
        chunk_text = "\n".join(current_lines)
        if len(chunk_text) >= min_chars:
            end_line = current_start + len(current_lines) - 1
            full_text = f"// File: {filepath}\n{chunk_text}"
            chunks.append(
                _make_chunk(
                    full_text,
                    filepath,
                    current_start,
                    end_line,
                    node_type,
                    language,
                )
            )

    return chunks


def _is_arrow_function_assignment(node) -> bool:
    """Return True if node is an arrow function assigned to a variable."""
    parent = node.parent
    if parent is None:
        return False
    if parent.type == "variable_declarator":
        return True
    return False


def _collect_chunks(
    node,
    source_bytes: bytes,
    filepath: str,
    language: str,
    visited_ids: set[int],
    target_nodes: frozenset[str],
    descend_into_export: bool = False,
) -> list[dict]:
    """Recursively walk the AST and collect chunks for target node types."""
    chunks: list[dict] = []

    if id(node) in visited_ids:
        return chunks

    node_type = node.type

    if node_type in target_nodes:
        # only capture arrow functions assigned to variables
        if node_type == "arrow_function" and not _is_arrow_function_assignment(node):
            # skip arrow_function itself; recurse into children
            for child in node.children:
                chunks.extend(
                    _collect_chunks(child, source_bytes, filepath, language, visited_ids, target_nodes, descend_into_export)
                )
            return chunks

        visited_ids.add(id(node))
        code = node.text.decode("utf-8") if node.text else ""
        start_line = node.start_point[0] + 1  # 1-based
        end_line = node.end_point[0] + 1

        full_text = f"// File: {filepath}\n{code}"

        if len(full_text) < config.CHUNK_MIN_CHARS:
            pass  # skip below min size
        elif len(full_text) > config.CHUNK_MAX_CHARS:
            chunks.extend(
                _split_long_chunk(
                    code,
                    filepath,
                    start_line,
                    node_type,
                    language,
                    config.CHUNK_MAX_CHARS,
                )
            )
        else:
            chunks.append(
                _make_chunk(full_text, filepath, start_line, end_line, node_type, language)
            )

        # descend into export_statement to capture named exports (TypeScript)
        if descend_into_export and node_type == "export_statement":
            for child in node.children:
                chunks.extend(
                    _collect_chunks(child, source_bytes, filepath, language, visited_ids, target_nodes, descend_into_export)
                )
        return chunks

    # recurse into children
    for child in node.children:
        chunks.extend(
            _collect_chunks(child, source_bytes, filepath, language, visited_ids, target_nodes, descend_into_export)
        )

    return chunks


def _collect_ts_chunks(
    node,
    source_bytes: bytes,
    filepath: str,
    language: str,
    visited_ids: set[int],
) -> list[dict]:
    """Walk TypeScript AST and collect chunks."""
    return _collect_chunks(
        node, source_bytes, filepath, language, visited_ids,
        target_nodes=TS_TARGET_NODES,
        descend_into_export=True,
    )


def _collect_go_chunks(
    node,
    source_bytes: bytes,
    filepath: str,
    language: str,
    visited_ids: set[int],
) -> list[dict]:
    """Walk Go AST and collect chunks."""
    return _collect_chunks(
        node, source_bytes, filepath, language, visited_ids,
        target_nodes=GO_TARGET_NODES,
        descend_into_export=False,
    )


# Chunker

class Chunker:
    """Split TypeScript/Go files into AST chunks using tree-sitter."""

    def __init__(self) -> None:
        self._ts_parser = _ts_parser
        self._tsx_parser = _tsx_parser
        self._go_parser = _go_parser

    def chunk_file(self, filepath: str) -> list[dict]:
        """Parse a single file and return a list of chunk dicts."""
        # extension check
        _, ext = os.path.splitext(filepath)
        if ext not in config.TARGET_EXTENSIONS:
            return []

        language = config.TARGET_EXTENSIONS[ext]

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                source_code = f.read()
        except OSError:
            return []

        source_bytes = source_code.encode("utf-8")

        # select parser
        if ext == ".tsx":
            parser = self._tsx_parser
        elif ext == ".ts":
            parser = self._ts_parser
        else:
            parser = self._go_parser

        tree = parser.parse(source_bytes)
        root_node = tree.root_node

        # normalize path separators
        normalized_path = filepath.replace("\\", "/")

        visited_ids: set[int] = set()

        if language == "typescript":
            return _collect_ts_chunks(
                root_node, source_bytes, normalized_path, language, visited_ids
            )
        elif language == "go":
            return _collect_go_chunks(
                root_node, source_bytes, normalized_path, language, visited_ids
            )
        else:
            return []

    def chunk_directory(self, dir_path: str) -> list[dict]:
        """Walk a directory and return chunks for all target files."""
        all_chunks: list[dict] = []

        for root, dirs, files in os.walk(dir_path):
            # prune excluded dirs in-place to prevent os.walk from descending
            dirs[:] = [
                d for d in dirs
                if d not in config.EXCLUDED_DIRS
            ]

            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext not in config.TARGET_EXTENSIONS:
                    continue

                full_path = os.path.join(root, filename)
                chunks = self.chunk_file(full_path)
                all_chunks.extend(chunks)

        return all_chunks
