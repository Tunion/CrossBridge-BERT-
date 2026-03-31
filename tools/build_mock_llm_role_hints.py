from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_jsonl
from common.llm_role_hint_utils import save_llm_role_hints


AUTH_RE = re.compile(r"\b(require|assert|verify|proof|signature|owner|admin|governor|quorum|validator)\b", re.I)
EXTERNAL_RE = re.compile(r"\b(call|delegatecall|transfer|send|router|bridge|execute|invoke|emit|message)\b", re.I)
STATE_RE = re.compile(r"\b(assign|mint|burn|lock|unlock|nonce|processed|balance|supply|claim|release|stored)\b", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mock LLM role hints from exported candidates for smoke tests.")
    p.add_argument("--input", type=str, default="outputs/results/llm_role_candidates.jsonl")
    p.add_argument("--output", type=str, default="outputs/results/llm_role_hints_mock.jsonl")
    p.add_argument("--max-items", type=int, default=64)
    return p.parse_args()


def resolve_path(path_text: str, *, must_exist: bool) -> Path:
    raw = Path(str(path_text).strip())
    candidates = [raw] if raw.is_absolute() else [ROOT / raw, raw]
    for cand in candidates:
        if not must_exist or cand.exists():
            return cand.resolve()
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve path: {path_text}. Tried: {tried}")


def clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def make_hint(row: Dict[str, Any]) -> Dict[str, Any]:
    fn = str(row.get("function_name", "") or "")
    preview = str(row.get("source_preview", "") or "")
    text = f"{fn}\n{preview}"
    auth = max(float(row.get("auth_prob_heuristic", 0.0)), 0.0)
    external = max(float(row.get("external_prob_heuristic", 0.0)), 0.0)
    state = max(float(row.get("state_prob_heuristic", 0.0)), 0.0)
    if AUTH_RE.search(text):
        auth = max(auth, 0.86)
    if EXTERNAL_RE.search(text):
        external = max(external, 0.86)
    if STATE_RE.search(text):
        state = max(state, 0.86)
    return {
        "relative_source_path": row.get("relative_source_path"),
        "function_name": row.get("function_name"),
        "auth_prob": clamp01(auth),
        "external_prob": clamp01(external),
        "state_prob": clamp01(state),
        "summary": "mock_llm_hint_for_smoke",
        "source": "mock-llm",
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input, must_exist=True)
    output_path = resolve_path(args.output, must_exist=False)
    rows: List[Dict[str, Any]] = list(load_jsonl(input_path))
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]
    hints = [make_hint(row) for row in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_llm_role_hints(output_path, hints)
    print(f"[build_mock_llm_role_hints] hints={len(hints)} input={input_path} output={output_path}")


if __name__ == "__main__":
    main()
