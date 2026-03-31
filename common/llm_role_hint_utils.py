from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from common.io_utils import load_jsonl, save_jsonl


ROLE_KEYS: Tuple[str, ...] = ("auth_constraint", "external_interaction", "state_change")


def canonical_function_name(name: Any) -> str:
    text = str(name or "").strip().lower()
    if not text:
        return "<unknown_fn>"
    return text.split("|", 1)[0].strip() or "<unknown_fn>"


def canonical_file_key(path_text: Any) -> str:
    text = str(path_text or "").replace("\\", "/").strip()
    return text or "<unknown_file>"


def make_hint_key(relative_source_path: Any, function_name: Any) -> str:
    return f"{canonical_file_key(relative_source_path)}:::{canonical_function_name(function_name)}"


def clamp01(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def normalize_hint_row(row: Dict[str, Any]) -> Dict[str, Any]:
    file_key = canonical_file_key(
        row.get("relative_source_path") or row.get("file") or row.get("source_path")
    )
    fn_key = canonical_function_name(row.get("function_name") or row.get("function"))
    out = {
        "relative_source_path": file_key,
        "function_name": fn_key,
        "auth_prob": clamp01(row.get("auth_prob", 0.0)),
        "external_prob": clamp01(row.get("external_prob", 0.0)),
        "state_prob": clamp01(row.get("state_prob", 0.0)),
        "summary": str(row.get("summary", "") or "").strip(),
        "source": str(row.get("source", "") or "").strip(),
    }
    out["hint_key"] = make_hint_key(file_key, fn_key)
    return out


def load_llm_role_hints(path: Path) -> Dict[str, Dict[str, Any]]:
    hints: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return hints
    for row in load_jsonl(path):
        norm = normalize_hint_row(dict(row))
        hints[str(norm["hint_key"])] = norm
    return hints


def save_llm_role_hints(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    save_jsonl(path, [normalize_hint_row(dict(r)) for r in rows])


def hint_probs(row: Dict[str, Any]) -> Dict[str, float]:
    return {
        "auth_constraint": clamp01(row.get("auth_prob", 0.0)),
        "external_interaction": clamp01(row.get("external_prob", 0.0)),
        "state_change": clamp01(row.get("state_prob", 0.0)),
    }

