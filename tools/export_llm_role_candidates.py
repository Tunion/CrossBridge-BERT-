from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import load_json, save_jsonl
from common.llm_role_hint_utils import ROLE_KEYS, canonical_function_name, canonical_file_key
from slicing.role_tagging import _role_confidence, score_node_roles


BRIDGE_KEYWORDS = (
    "bridge", "router", "endpoint", "message", "payload", "nonce", "proof", "verify",
    "mint", "burn", "lock", "unlock", "claim", "release", "execute", "send", "receive",
    "lz", "wormhole", "axelar", "layerzero", "omni",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export low-confidence function candidates for LLM role hints.")
    p.add_argument("--graph-dir", type=str, default="data/graphs/full")
    p.add_argument("--output", type=str, default="outputs/results/llm_role_candidates.jsonl")
    p.add_argument("--max-graphs", type=int, default=0)
    p.add_argument("--low-confidence-max", type=float, default=0.60)
    p.add_argument("--max-functions", type=int, default=0)
    p.add_argument("--min-preview-chars", type=int, default=80)
    p.add_argument("--allow-empty-preview", action="store_true")
    return p.parse_args()


def candidate_priority(fn_name: str, contract_name: str, preview: str, scores: Dict[str, float], conf: float) -> float:
    text = " ".join([fn_name, contract_name, preview]).lower()
    keyword_hits = sum(1 for kw in BRIDGE_KEYWORDS if kw in text)
    role_peak = max(float(scores.get(k, 0.0)) for k in ROLE_KEYS)
    preview_bonus = min(len(preview) / 400.0, 1.0)
    low_conf_bonus = 1.0 - float(conf)
    return 2.0 * role_peak + 0.35 * keyword_hits + 0.25 * preview_bonus + 0.40 * low_conf_bonus


def main() -> None:
    args = parse_args()
    graph_dir = (ROOT / args.graph_dir).resolve()
    out_path = (ROOT / args.output).resolve()
    paths = sorted(graph_dir.glob("*.json"))
    if args.max_graphs and args.max_graphs > 0:
        paths = paths[: args.max_graphs]

    rows: List[Dict[str, Any]] = []
    for p in paths:
        graph = load_json(p)
        file_key = canonical_file_key(graph.get("meta", {}).get("relative_source_path", p.name))
        fn_scores: Dict[str, Dict[str, float]] = {}
        fn_texts: Dict[str, List[str]] = {}
        fn_contracts: Dict[str, str] = {}
        for node in graph.get("nodes", []):
            scores, _ = score_node_roles(node)
            fn_name = canonical_function_name(node.get("function") or node.get("name"))
            if fn_name == "<unknown_fn>":
                continue
            agg = fn_scores.setdefault(fn_name, {k: 0.0 for k in ROLE_KEYS})
            for rk in ROLE_KEYS:
                agg[rk] = max(float(agg[rk]), float(scores.get(rk, 0.0)))
            node_type = str(node.get("type", "") or "").lower()
            text = str(node.get("text", "") or "").strip()
            if text:
                if node_type in {"statement", "condition", "call"}:
                    fn_texts.setdefault(fn_name, []).append(text)
            contract_name = str(node.get("contract_name", "") or "").strip()
            if contract_name and fn_name not in fn_contracts:
                fn_contracts[fn_name] = contract_name
        for fn_name, scores in fn_scores.items():
            conf = _role_confidence(scores)
            if conf > float(args.low_confidence_max):
                continue
            preview = "\n".join(fn_texts.get(fn_name, [])[:12])[:4000]
            if not args.allow_empty_preview and len(preview.strip()) < int(args.min_preview_chars):
                continue
            contract_name = fn_contracts.get(fn_name, "")
            prompt = (
                "You are labeling a Solidity function for bridge-mechanism role hints.\n"
                "Return exactly one strict JSON object with keys auth_prob, external_prob, state_prob, summary.\n"
                "Probabilities must be numeric in [0,1]. Never return null.\n\n"
                "Role rubric:\n"
                "- auth_prob: non-zero if the function performs or directly participates in authorization, proof/signature verification, approval gating, command validation, or processed/approved checks.\n"
                "- external_prob: non-zero if the function emits bridge-facing events, performs external calls, invokes endpoints/gateways/routers, or participates in cross-domain message execution.\n"
                "- state_prob: non-zero if the function writes approval/processed/nonces/messages/balances/locks/releases or updates bridge state.\n"
                "- Functions named approve, verify, execute, mint, release, unlock, setApproved, setProcessed, consume, validate often imply non-zero bridge roles.\n"
                "- Hash/key builder helpers that only compute keccak/abi.encode without approval checks, external interaction, or state writes should usually stay near zero.\n\n"
                "Few-shot guidance:\n"
                'Example A input: function approveMessage(...) { processed[messageId] = true; emit MessageApproved(messageId); }\n'
                'Example A output: {"auth_prob":0.75,"external_prob":0.55,"state_prob":0.85,"summary":"Approves a bridge message, emits an approval event, and updates processed state."}\n'
                'Example B input: function computeMessageKey(...) returns (bytes32) { return keccak256(abi.encode(...)); }\n'
                'Example B output: {"auth_prob":0.0,"external_prob":0.0,"state_prob":0.0,"summary":"Pure key/hash helper without authorization, external interaction, or state update."}\n\n'
                f"File: {file_key}\nFunction: {fn_name}\nContract: {contract_name}\n\n"
                f"Function excerpt:\n{preview}"
            )
            rows.append(
                {
                    "relative_source_path": file_key,
                    "function_name": fn_name,
                    "contract_name": contract_name,
                    "auth_prob_heuristic": float(scores.get("auth_constraint", 0.0)),
                    "external_prob_heuristic": float(scores.get("external_interaction", 0.0)),
                    "state_prob_heuristic": float(scores.get("state_change", 0.0)),
                    "heuristic_confidence": float(conf),
                    "candidate_priority": float(candidate_priority(fn_name, contract_name, preview, scores, conf)),
                    "source_preview": preview,
                    "prompt": prompt,
                }
            )
    rows.sort(key=lambda r: (-float(r.get("candidate_priority", 0.0)), r["relative_source_path"], r["function_name"]))
    if args.max_functions and args.max_functions > 0:
        rows = rows[: args.max_functions]
    save_jsonl(out_path, rows)
    print(f"[export_llm_role_candidates] candidates={len(rows)} output={out_path}")


if __name__ == "__main__":
    main()
