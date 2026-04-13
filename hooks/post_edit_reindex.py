"""
Claude Code PostToolUse hook. Incrementally re-indexes .ts/.tsx/.go files after Edit/Write/MultiEdit.

stdin: {"tool_name": "...", "tool_input": {"file_path": "...", ...}}
"""

import json
import os
import subprocess
import sys
from pathlib import Path

TARGET_EXTENSIONS = {".ts", ".tsx", ".go"}
PROJECT_DIR = r"C:\Users\naxpc\Devs\GoLand\mix-share"

PYTHON = str(
    Path(__file__).parent.parent / ".venv" / "Scripts" / "python.exe"
)
CLI = str(Path(__file__).parent.parent / "cli.py")


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_input = payload.get("tool_input", {})

    # Edit/Write use file_path; MultiEdit uses edits[].file_path
    file_paths: list[str] = []
    if "file_path" in tool_input:
        file_paths.append(tool_input["file_path"])
    elif "edits" in tool_input:
        file_paths.extend(e.get("file_path", "") for e in tool_input.get("edits", []))

    # check if any target extensions modified
    needs_reindex = any(
        Path(p).suffix.lower() in TARGET_EXTENSIONS for p in file_paths if p
    )

    if not needs_reindex:
        sys.exit(0)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    result = subprocess.run(
        [PYTHON, CLI, "update", PROJECT_DIR],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[reindex] Error: {result.stderr}", file=sys.stderr)
    else:
        # only print if files were updated
        if "files_updated=0" not in result.stdout:
            print(f"[reindex] {result.stdout.strip()}", file=sys.stderr)


if __name__ == "__main__":
    main()
