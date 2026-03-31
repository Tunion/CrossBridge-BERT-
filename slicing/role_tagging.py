from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, save_json
from common.llm_role_hint_utils import (
    ROLE_KEYS,
    canonical_function_name,
    load_llm_role_hints,
    make_hint_key,
)


STATE_KEYWORDS = ("mint", "burn", "lock", "unlock", "nonce", "balance", "allowance", "supply")
EXTERNAL_KEYWORDS = ("call", "delegatecall", "transfer", "send", "router", "bridge", "invoke", "execute")
AUTH_KEYWORDS = ("require", "assert", "onlyowner", "onlyadmin", "signature", "proof", "msg.sender", "verify")
ASSIGN_PATTERN = re.compile(r"(^|[^=!<>])=(?!=)")
MUTATION_OPS = ("+=", "-=", "*=", "/=", "%=", "++", "--")


def _empty_role_scores() -> Dict[str, float]:
    return {k: 0.0 for k in ROLE_KEYS}


def _score_increment(scores: Dict[str, float], key: str, value: float) -> None:
    if key in scores:
        scores[key] += float(value)


def semantic_tags(node: Dict[str, Any]) -> List[str]:
    vals: List[str] = []
    sem = node.get("semantic_class")
    if isinstance(sem, str) and sem:
        vals.append(sem.lower())
    sem_all = node.get("semantic_all", [])
    if isinstance(sem_all, list):
        vals.extend(str(x).lower() for x in sem_all if isinstance(x, str))
    return vals


def has_assignment(text: str) -> bool:
    if any(op in text for op in MUTATION_OPS):
        return True
    return ASSIGN_PATTERN.search(text) is not None


def score_node_roles(node: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    scores = _empty_role_scores()
    node_type = str(node.get("type", ""))
    sem_tags = semantic_tags(node)
    sem_text = " ".join(sem_tags)
    text = str(node.get("text", "")).lower()
    stmt_type = str(node.get("stmt_type", "")).lower()

    if node_type == "state_var":
        _score_increment(scores, "state_change", 1.00)
    if node_type == "call":
        _score_increment(scores, "external_interaction", 1.00)
    if node_type == "condition":
        _score_increment(scores, "auth_constraint", 1.00)

    if "state_change" in sem_text:
        _score_increment(scores, "state_change", 0.90)
    if "external_interaction" in sem_text:
        _score_increment(scores, "external_interaction", 0.90)
    if "constraint_check" in sem_text:
        _score_increment(scores, "auth_constraint", 0.90)

    # Heuristic fallback only for statement-like nodes.
    if node_type in {"statement", "condition", "call"}:
        if stmt_type in {"call", "emit"} or any(k in text for k in EXTERNAL_KEYWORDS):
            _score_increment(scores, "external_interaction", 0.55)
        if stmt_type in {"constraint", "if"} or any(k in text for k in AUTH_KEYWORDS):
            _score_increment(scores, "auth_constraint", 0.55)
        if stmt_type in {"assign", "var_decl"} and (
            has_assignment(text) or any(k in text for k in STATE_KEYWORDS)
        ):
            _score_increment(scores, "state_change", 0.55)

    # Filter accidental tags for comments/context.
    if node_type == "statement" and node.get("stmt_type") == "comment":
        return _empty_role_scores(), []
    roles = [k for k, v in scores.items() if float(v) >= 0.80]
    return scores, sorted(set(roles))


def _role_confidence(scores: Dict[str, float]) -> float:
    vals = sorted([float(scores.get(k, 0.0)) for k in ROLE_KEYS], reverse=True)
    if not vals:
        return 0.0
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    return float(max(0.0, min(1.0, top1 - 0.35 * top2)))


def _node_function_name(node: Dict[str, Any]) -> str:
    return canonical_function_name(node.get("function") or node.get("name"))


def _compatible_for_hint(node: Dict[str, Any], role: str) -> bool:
    node_type = str(node.get("type", "")).lower()
    stmt_type = str(node.get("stmt_type", "")).lower()
    text = str(node.get("text", "")).lower()
    if role == "auth_constraint":
        return node_type == "condition" or stmt_type in {"if", "constraint"} or any(k in text for k in AUTH_KEYWORDS)
    if role == "external_interaction":
        return node_type == "call" or stmt_type in {"call", "emit"} or any(k in text for k in EXTERNAL_KEYWORDS)
    if role == "state_change":
        return node_type == "state_var" or (
            stmt_type in {"assign", "var_decl"} and (has_assignment(text) or any(k in text for k in STATE_KEYWORDS))
        )
    return False


def process_graph(
    graph: Dict[str, Any],
    llm_hints: Dict[str, Dict[str, Any]] | None = None,
    llm_low_conf_max: float = 0.60,
    llm_min_prob: float = 0.72,
    llm_weight: float = 0.45,
) -> Dict[str, Any]:
    llm_hints = llm_hints or {}
    fn_nodes: Dict[str, List[int]] = collections.defaultdict(list)
    fn_agg = collections.defaultdict(_empty_role_scores)
    llm_hint_function_count = 0
    llm_hint_node_updates = 0
    for node in graph.get("nodes", []):
        scores, roles = score_node_roles(node)
        node["role_scores"] = {k: float(scores.get(k, 0.0)) for k in ROLE_KEYS}
        node["role_confidence"] = _role_confidence(scores)
        node["roles"] = roles
        fn_name = _node_function_name(node)
        if fn_name and fn_name != "<unknown_fn>":
            fn_nodes[fn_name].append(int(node.get("id", -1)))
            for rk in ROLE_KEYS:
                fn_agg[fn_name][rk] = max(float(fn_agg[fn_name][rk]), float(scores.get(rk, 0.0)))

    id_to_node = {int(n.get("id", -1)): n for n in graph.get("nodes", [])}
    file_key = str(graph.get("meta", {}).get("relative_source_path", "") or "")
    for fn_name, node_ids in fn_nodes.items():
        fn_scores = fn_agg[fn_name]
        fn_conf = _role_confidence(fn_scores)
        hint = llm_hints.get(make_hint_key(file_key, fn_name))
        if not hint or fn_conf > float(llm_low_conf_max):
            continue
        llm_hint_function_count += 1
        hint_scores = {
            "auth_constraint": float(hint.get("auth_prob", 0.0)),
            "external_interaction": float(hint.get("external_prob", 0.0)),
            "state_change": float(hint.get("state_prob", 0.0)),
        }
        for node_id in node_ids:
            node = id_to_node.get(int(node_id))
            if not node:
                continue
            merged = dict(node.get("role_scores") or _empty_role_scores())
            changed = False
            for rk in ROLE_KEYS:
                hval = float(hint_scores.get(rk, 0.0))
                if hval < float(llm_min_prob):
                    continue
                if not _compatible_for_hint(node, rk):
                    continue
                merged[rk] = max(float(merged.get(rk, 0.0)), float(llm_weight) * hval)
                if rk not in (node.get("roles") or []):
                    changed = True
            node["role_scores"] = {k: float(merged.get(k, 0.0)) for k in ROLE_KEYS}
            node["role_confidence"] = _role_confidence(merged)
            node["llm_role_hint_used"] = bool(changed)
            node["llm_role_hint_source"] = str(hint.get("source", "") or "hint")
            node["llm_role_hint_summary"] = str(hint.get("summary", "") or "")
            node["roles"] = sorted({k for k, v in merged.items() if float(v) >= 0.80} | set(node.get("roles", []) or []))
            if changed:
                llm_hint_node_updates += 1
    graph.setdefault("meta", {})
    role_nodes = sum(1 for n in graph.get("nodes", []) if n.get("roles"))
    graph["meta"]["role_node_count"] = role_nodes
    graph["meta"]["llm_role_hint_enabled"] = bool(llm_hints)
    graph["meta"]["llm_role_hint_function_count"] = int(llm_hint_function_count)
    graph["meta"]["llm_role_hint_node_updates"] = int(llm_hint_node_updates)
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2: role tagging on top of hetero graph.")
    parser.add_argument("--graph-dir", type=str, default="data/graphs/full")
    parser.add_argument("--output-dir", type=str, default="data/graphs/tagged")
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--llm-role-hints", type=str, default="")
    parser.add_argument("--llm-role-low-confidence-max", type=float, default=0.60)
    parser.add_argument("--llm-role-min-prob", type=float, default=0.72)
    parser.add_argument("--llm-role-weight", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = (ROOT / args.graph_dir).resolve()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(graph_dir.glob("*.json"), key=lambda x: x.name)
    if args.max_graphs and args.max_graphs > 0:
        paths = paths[: args.max_graphs]

    llm_hints: Dict[str, Dict[str, Any]] = {}
    if str(args.llm_role_hints).strip():
        llm_hints = load_llm_role_hints((ROOT / args.llm_role_hints).resolve())

    for p in paths:
        g = load_json(p)
        g = process_graph(
            g,
            llm_hints=llm_hints,
            llm_low_conf_max=float(args.llm_role_low_confidence_max),
            llm_min_prob=float(args.llm_role_min_prob),
            llm_weight=float(args.llm_role_weight),
        )
        save_json(out_dir / p.name, g)
    print(f"[role_tagging] graphs={len(paths)} output_dir={out_dir}")


if __name__ == "__main__":
    main()
