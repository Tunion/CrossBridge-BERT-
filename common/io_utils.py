from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_text(path: Path, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding, errors="ignore")


def write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    ensure_parent(path)
    path.write_text(text, encoding=encoding)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_posix(path: Path) -> str:
    return path.as_posix()


def resolve_solidity_paths(
    project_root: Path,
    dataset_root: Path,
    input_list: Optional[Path] = None,
    glob_pattern: str = "**/*.sol",
    max_files: int = 0,
) -> List[Path]:
    paths: List[Path] = []
    if input_list is not None and input_list.exists():
        for line in input_list.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute():
                p = project_root / line.replace("/", "\\")
            if p.exists() and p.suffix.lower() == ".sol":
                paths.append(p.resolve())
    else:
        for p in dataset_root.rglob(glob_pattern):
            if p.is_file() and p.suffix.lower() == ".sol":
                paths.append(p.resolve())
    paths = sorted(set(paths), key=lambda x: str(x))
    if max_files and max_files > 0:
        paths = paths[:max_files]
    return paths

