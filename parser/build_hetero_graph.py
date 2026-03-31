from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph_utils import add_edge, add_node, dedup_edges, init_graph
from common.io_utils import load_jsonl, save_json, save_jsonl


DATA_OBJECT_KEYS = (
    "amount",
    "payload",
    "proof",
    "signature",
    "nonce",
    "balance",
    "recipient",
    "sender",
    "token",
    "fee",
    "chainid",
    "bridge",
    "router",
)

INTERPROC_OBJECT_KEYS = {
    "amount",
    "payload",
    "proof",
    "signature",
    "nonce",
    "token",
    "chainid",
    "bridge",
    "router",
}

IMPLICIT_MESSAGE_KEYS = {
    "payload",
    "proof",
    "signature",
    "nonce",
    "recipient",
    "sender",
    "token",
    "amount",
    "chainid",
}
VERIFY_HINTS = (
    "verify",
    "proof",
    "signature",
    "checkpoint",
    "validator",
    "quorum",
    "root",
    "merkle",
)
EXECUTE_HINTS = (
    "execute",
    "unlock",
    "release",
    "claim",
    "mint",
    "receive",
    "fulfill",
    "redeem",
)
STATE_LANDING_HINTS = (
    "nonce",
    "processed",
    "claimed",
    "balance",
    "mint",
    "burn",
    "lock",
    "release",
    "unlock",
    "claim",
)

AST_EDGE_TYPES = {"ast_parent"}
CFG_EDGE_TYPES = {"cfg_next", "control_dep"}
DFG_EDGE_TYPES = {"dfg_dep", "data_dep"}
MAX_INTERPROC_STATE_PAIRS_PER_VAR = 24

ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_\.\[\]]*)\s*[\+\-\*\/]?=")
VAR_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
COMMENT_LINE_RE = re.compile(r"^\s*(//|/\*|\*|\*/)")

SOLIDITY_KEYWORDS = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "return",
    "returns",
    "function",
    "modifier",
    "constructor",
    "fallback",
    "receive",
    "event",
    "emit",
    "mapping",
    "address",
    "uint",
    "uint256",
    "uint128",
    "uint64",
    "uint32",
    "uint16",
    "uint8",
    "int",
    "int256",
    "bool",
    "bytes",
    "bytes32",
    "string",
    "memory",
    "storage",
    "calldata",
    "public",
    "private",
    "internal",
    "external",
    "view",
    "pure",
    "payable",
    "virtual",
    "override",
    "constant",
    "immutable",
    "new",
    "delete",
    "require",
    "assert",
    "revert",
    "true",
    "false",
    "this",
    "msg",
    "sender",
    "value",
    "data",
    "tx",
    "origin",
    "block",
    "timestamp",
    "number",
}


def extract_data_objects(text: str) -> List[str]:
    low = text.lower()
    return sorted({k for k in DATA_OBJECT_KEYS if k in low})


def extract_state_vars(text: str) -> List[str]:
    out = []
    for m in ASSIGN_RE.finditer(text):
        lhs = m.group(1)
        # Skip comparisons-like false positives.
        if lhs in {"require", "assert"}:
            continue
        out.append(lhs)
    return sorted(set(out))


def normalize_ident(x: str) -> str:
    y = x.split("[", 1)[0].split(".", 1)[0].strip()
    return y


def normalize_call_target(name: str) -> str:
    x = str(name or "").strip()
    if not x:
        return ""
    x = x.split("(", 1)[0].strip()
    if "." in x:
        x = x.rsplit(".", 1)[-1]
    return normalize_ident(x)


def pick_representative_ids(items: Sequence[int], limit: int = 1) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for x in sorted(int(v) for v in items):
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
        if len(out) >= max(1, int(limit)):
            break
    return out


def is_comment_stmt(stmt: Dict[str, Any]) -> bool:
    stmt_type = str(stmt.get("stmt_type", "")).strip().lower()
    if stmt_type == "comment":
        return True
    text = str(stmt.get("text", "")).strip()
    return bool(COMMENT_LINE_RE.match(text))


def collect_known_state_vars(record: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for fn in record.get("functions", []):
        for stmt in fn.get("statements", []):
            for x in stmt.get("state_defs", []) or []:
                if not isinstance(x, str):
                    continue
                y = normalize_ident(x)
                if y:
                    out.add(y)
    return out


def bfs_reachable_risk_targets(
    start: int,
    cfg_adj: Dict[int, Set[int]],
    risk_nodes: Set[int],
    max_hops: int = 3,
    max_targets: int = 2,
) -> List[Tuple[int, int]]:
    if start < 0 or not risk_nodes:
        return []
    q: Deque[Tuple[int, int]] = collections.deque([(start, 0)])
    seen: Set[int] = {start}
    found: List[Tuple[int, int]] = []
    while q:
        cur, dep = q.popleft()
        if dep >= max_hops:
            continue
        nd = dep + 1
        for nb in cfg_adj.get(cur, set()):
            if nb in seen:
                continue
            seen.add(nb)
            if nb in risk_nodes:
                found.append((nb, nd))
                if len(found) >= max_targets:
                    return found
            q.append((nb, nd))
    return found


def extract_defs_uses(text: str) -> Tuple[List[str], List[str]]:
    defs = []
    for m in ASSIGN_RE.finditer(text):
        lhs = normalize_ident(m.group(1))
        if not lhs:
            continue
        if lhs.lower() in SOLIDITY_KEYWORDS:
            continue
        defs.append(lhs)
    def_set = set(defs)

    uses = []
    for tok in VAR_TOKEN_RE.findall(text):
        if tok.lower() in SOLIDITY_KEYWORDS:
            continue
        if tok in def_set:
            continue
        uses.append(tok)
    return sorted(def_set), sorted(set(uses))


def stmt_text_lower(stmt: Dict[str, Any]) -> str:
    return str(stmt.get("text", "") or "").strip().lower()


def stmt_message_keys(stmt: Dict[str, Any]) -> Set[str]:
    keys = {str(x).strip().lower() for x in (stmt.get("data_objects", []) or []) if isinstance(x, str)}
    if not keys:
        keys = set(extract_data_objects(str(stmt.get("text", "") or "")))
    return {k for k in keys if k in IMPLICIT_MESSAGE_KEYS}


def is_emit_stmt(stmt: Dict[str, Any]) -> bool:
    st = str(stmt.get("stmt_type", "")).strip().lower()
    txt = stmt_text_lower(stmt)
    return st == "emit" or txt.startswith("emit ")


def is_verify_like_stmt(stmt: Dict[str, Any]) -> bool:
    txt = stmt_text_lower(stmt)
    sem = {str(x).strip().lower() for x in (stmt.get("semantic_class", []) or [])}
    has_verify_hint = any(k in txt for k in VERIFY_HINTS)
    has_msg_key = bool(stmt_message_keys(stmt))
    return ("constraint_check" in sem or "auth_constraint" in sem or "auth" in sem) and (has_verify_hint or has_msg_key)


def is_execute_like_stmt(stmt: Dict[str, Any]) -> bool:
    txt = stmt_text_lower(stmt)
    sem = {str(x).strip().lower() for x in (stmt.get("semantic_class", []) or [])}
    if "external_interaction" in sem and (any(k in txt for k in EXECUTE_HINTS) or bool(stmt_message_keys(stmt))):
        return True
    if "state_change" in sem and any(k in txt for k in EXECUTE_HINTS):
        return True
    return False


def is_state_landing_stmt(stmt: Dict[str, Any]) -> bool:
    txt = stmt_text_lower(stmt)
    sem = {str(x).strip().lower() for x in (stmt.get("semantic_class", []) or [])}
    has_state_defs = bool(stmt.get("state_defs", []) or [])
    return "state_change" in sem and (has_state_defs or any(k in txt for k in STATE_LANDING_HINTS))


def build_graph(record: Dict[str, Any]) -> Dict[str, Any]:
    g = init_graph(
        contract_id=record["contract_id"],
        source_path=record["source_path"],
        file_name=record["file_name"],
    )
    g["meta"]["relative_source_path"] = record.get("relative_source_path")
    g["meta"]["line_count"] = record.get("line_count", 0)
    g["meta"]["parser_backend"] = record.get("parser_backend")
    g["meta"]["ast_parse_ok"] = record.get("ast_parse_ok")

    data_obj_nodes: Dict[str, int] = {}
    state_var_nodes: Dict[str, int] = {}
    fn_name_to_id: Dict[str, int] = {}
    fn_summaries: Dict[str, Dict[str, Any]] = {}
    implicit_mechanism_edges_total = 0
    known_state_vars = collect_known_state_vars(record)

    for fn in record.get("functions", []):
        fn_name = str(fn.get("name", ""))
        fn_key = fn_name.lower() if fn_name else f"<anonymous_function_{len(fn_summaries)}>"
        fn_id = add_node(
            g,
            "function",
            {
                "name": fn_name,
                "contract_name": fn.get("contract_name"),
                "start_line": fn.get("start_line"),
                "end_line": fn.get("end_line"),
                "modifiers": fn.get("modifiers", []),
            },
        )
        if fn_name:
            fn_name_to_id[fn_key] = fn_id
        fn_summaries[fn_key] = {
            "fn_id": fn_id,
            "fn_name": fn_name,
            "local_calls": set(),
            "state_write_vars": set(),
            "state_read_vars": set(),
            "write_stmt_by_var": collections.defaultdict(list),
            "read_stmt_by_var": collections.defaultdict(list),
            "object_stmt_by_key": collections.defaultdict(list),
        }

    for fn in record.get("functions", []):
        fn_name = str(fn.get("name", ""))
        fn_key = fn_name.lower() if fn_name else ""
        fn_id = fn_name_to_id.get(fn_key)
        if fn_id is None:
            continue
        fn_summary = fn_summaries[fn_key]

        stmt_rows: List[Dict[str, Any]] = []
        stmt_idx_to_node: Dict[int, int] = {}
        stmt_idx_to_stmt: Dict[int, Dict[str, Any]] = {}
        last_def_stmt: Dict[str, int] = {}
        cfg_adj: Dict[int, Set[int]] = collections.defaultdict(set)
        emit_stmt_ids: List[Tuple[int, Set[str]]] = []
        verify_stmt_ids: List[Tuple[int, Set[str]]] = []
        execute_stmt_ids: List[Tuple[int, Set[str]]] = []
        state_landing_stmt_ids: List[Tuple[int, Set[str]]] = []
        for pos, stmt in enumerate(fn.get("statements", [])):
            if is_comment_stmt(stmt):
                continue
            stmt_idx = stmt.get("index")
            if not isinstance(stmt_idx, int):
                stmt_idx = pos
            sem_list = stmt.get("semantic_class", ["context"])
            sem = sem_list[0] if sem_list else "context"
            st_id = add_node(
                g,
                "statement",
                {
                    "function": fn_name,
                    "contract_name": fn.get("contract_name"),
                    "stmt_index": stmt_idx,
                    "line": stmt.get("line"),
                    "end_line": stmt.get("end_line", stmt.get("line")),
                    "stmt_type": stmt.get("stmt_type", "statement"),
                    "semantic_class": sem,
                    "semantic_all": sem_list,
                    "ast_type": stmt.get("ast_type"),
                    "text": stmt.get("text", ""),
                },
            )
            add_edge(g, fn_id, st_id, "contains")
            add_edge(g, fn_id, st_id, "ast_parent")
            stmt_rows.append({"node_id": st_id, "stmt": stmt, "idx": stmt_idx})
            stmt_idx_to_node[stmt_idx] = st_id
            stmt_idx_to_stmt[stmt_idx] = stmt
            msg_keys = stmt_message_keys(stmt)
            if is_emit_stmt(stmt):
                emit_stmt_ids.append((st_id, msg_keys))
            if is_verify_like_stmt(stmt):
                verify_stmt_ids.append((st_id, msg_keys))
            if is_execute_like_stmt(stmt):
                execute_stmt_ids.append((st_id, msg_keys))
            if is_state_landing_stmt(stmt):
                state_landing_stmt_ids.append((st_id, msg_keys))

            # Build explicit interaction node.
            call_targets = [x for x in stmt.get("call_targets", []) if isinstance(x, str)]
            if "external_interaction" in sem_list or call_targets:
                call_id = add_node(
                    g,
                    "call",
                    {
                        "function": fn_name,
                        "contract_name": fn.get("contract_name"),
                        "line": stmt.get("line"),
                        "semantic_class": "external_interaction",
                        "call_targets": call_targets,
                    },
                )
                add_edge(g, st_id, call_id, "call_interaction")
                for ct in call_targets:
                    c_name = normalize_call_target(ct).lower()
                    if not c_name:
                        continue
                    callee_fn_id = fn_name_to_id.get(c_name)
                    if callee_fn_id is None:
                        continue
                    add_edge(g, call_id, callee_fn_id, "call_interaction", {"kind": "local_call"})
                    fn_summary["local_calls"].add(callee_fn_id)

            # Build condition node for explicit constraints.
            if "constraint_check" in sem_list or stmt.get("stmt_type") in {"if", "constraint"}:
                cond_id = add_node(
                    g,
                    "condition",
                    {
                        "function": fn_name,
                        "contract_name": fn.get("contract_name"),
                        "line": stmt.get("line"),
                        "semantic_class": "constraint_check",
                    },
                )
                add_edge(g, st_id, cond_id, "constraint_on")

            # State vars referenced by writes.
            state_defs = [x for x in stmt.get("state_defs", []) if isinstance(x, str)]
            defs_from_stmt = [x for x in stmt.get("defs", []) if isinstance(x, str)]
            state_var_candidates = set(state_defs)
            if not state_var_candidates and "state_change" in sem_list:
                state_var_candidates.update(defs_from_stmt)
            if not state_var_candidates and "state_change" in sem_list:
                state_var_candidates.update(extract_state_vars(stmt.get("text", "")))
            for sv in sorted(state_var_candidates):
                sv_key = normalize_ident(sv)
                if not sv_key:
                    continue
                if known_state_vars and sv_key not in known_state_vars and "state_change" not in sem_list:
                    continue
                if sv_key not in state_var_nodes:
                    state_var_nodes[sv_key] = add_node(
                        g,
                        "state_var",
                        {"name": sv_key, "semantic_class": "state_change"},
                    )
                add_edge(g, st_id, state_var_nodes[sv_key], "data_dep")
                add_edge(g, st_id, state_var_nodes[sv_key], "dfg_dep")
                fn_summary["state_write_vars"].add(sv_key)
                fn_summary["write_stmt_by_var"][sv_key].append(st_id)

            # Key data objects.
            data_objects = [x for x in stmt.get("data_objects", []) if isinstance(x, str)]
            if not data_objects:
                data_objects = extract_data_objects(stmt.get("text", ""))
            for obj in sorted(set(data_objects)):
                if obj not in data_obj_nodes:
                    data_obj_nodes[obj] = add_node(
                        g,
                        "data_object",
                        {"name": obj, "semantic_class": "context"},
                    )
                add_edge(g, st_id, data_obj_nodes[obj], "data_dep")
                if obj in INTERPROC_OBJECT_KEYS:
                    fn_summary["object_stmt_by_key"][obj].append(st_id)

            # DFG: connect use sites to nearest previous def sites in the same function.
            defs = [normalize_ident(x) for x in stmt.get("defs", []) if isinstance(x, str)]
            uses = [normalize_ident(x) for x in stmt.get("uses", []) if isinstance(x, str)]
            defs = [x for x in defs if x]
            uses = [x for x in uses if x]
            if not defs and not uses:
                defs_fallback, uses_fallback = extract_defs_uses(stmt.get("text", ""))
                defs = [normalize_ident(x) for x in defs_fallback if normalize_ident(x)]
                uses = [normalize_ident(x) for x in uses_fallback if normalize_ident(x)]
            for var in uses:
                prev = last_def_stmt.get(var)
                if prev is not None and prev != st_id:
                    add_edge(g, prev, st_id, "dfg_dep", {"var": var})
                if var in known_state_vars:
                    sv_id = state_var_nodes.get(var)
                    if sv_id is None:
                        sv_id = add_node(
                            g,
                            "state_var",
                            {"name": var, "semantic_class": "state_change"},
                        )
                        state_var_nodes[var] = sv_id
                    add_edge(g, sv_id, st_id, "data_dep")
                    add_edge(g, sv_id, st_id, "dfg_dep", {"var": var, "kind": "state_read"})
                    fn_summary["state_read_vars"].add(var)
                    fn_summary["read_stmt_by_var"][var].append(st_id)
            for var in defs:
                last_def_stmt[var] = st_id

        # Statement-level AST parent (statement nesting).
        for row in stmt_rows:
            stmt = row["stmt"]
            child_id = row["node_id"]
            parent_idx = stmt.get("parent_stmt_index")
            if isinstance(parent_idx, int) and parent_idx in stmt_idx_to_node:
                add_edge(g, stmt_idx_to_node[parent_idx], child_id, "ast_parent")
                add_edge(g, stmt_idx_to_node[parent_idx], child_id, "control_dep")

        # CFG/control edges: prefer structured cfg_succ_indices from AST preprocessing.
        has_structured_cfg = any(isinstance(r["stmt"].get("cfg_succ_indices"), list) and r["stmt"].get("cfg_succ_indices") for r in stmt_rows)
        if has_structured_cfg:
            for row in stmt_rows:
                src_id = row["node_id"]
                src_stmt = row["stmt"]
                for succ_idx in src_stmt.get("cfg_succ_indices", []) or []:
                    if not isinstance(succ_idx, int):
                        continue
                    dst_id = stmt_idx_to_node.get(succ_idx)
                    if dst_id is None:
                        continue
                    dst_stmt = stmt_idx_to_stmt.get(succ_idx, {})
                    add_edge(
                        g,
                        src_id,
                        dst_id,
                        "control_dep",
                        {"line_from": src_stmt.get("line"), "line_to": dst_stmt.get("line"), "kind": "structured_cfg"},
                    )
                    add_edge(
                        g,
                        src_id,
                        dst_id,
                        "cfg_next",
                        {"line_from": src_stmt.get("line"), "line_to": dst_stmt.get("line"), "kind": "structured_cfg"},
                    )
                    cfg_adj[src_id].add(dst_id)
        else:
            for a, b in zip(stmt_rows, stmt_rows[1:]):
                a_id = a["node_id"]
                b_id = b["node_id"]
                a_stmt = a["stmt"]
                b_stmt = b["stmt"]
                add_edge(
                    g,
                    a_id,
                    b_id,
                    "control_dep",
                    {"line_from": a_stmt.get("line"), "line_to": b_stmt.get("line"), "kind": "sequential"},
                )
                add_edge(
                    g,
                    a_id,
                    b_id,
                    "cfg_next",
                    {"line_from": a_stmt.get("line"), "line_to": b_stmt.get("line"), "kind": "sequential"},
                )
                cfg_adj[a_id].add(b_id)

        # Constraint to nearby risky statements via local CFG reachability.
        constraints = [
            row["node_id"]
            for row in stmt_rows
            for s in [row["stmt"]]
            if "constraint_check" in s.get("semantic_class", [])
            or s.get("stmt_type") in {"if", "constraint"}
        ]
        risk_targets = [
            row["node_id"]
            for row in stmt_rows
            for s in [row["stmt"]]
            if any(x in s.get("semantic_class", []) for x in ("external_interaction", "state_change"))
        ]
        risk_set = set(risk_targets)
        for c_id in constraints:
            linked = bfs_reachable_risk_targets(
                start=c_id,
                cfg_adj=cfg_adj,
                risk_nodes=risk_set,
                max_hops=3,
                max_targets=2,
            )
            for t_id, hops in linked:
                if c_id == t_id:
                    continue
                add_edge(g, c_id, t_id, "constraint_on", {"kind": "cfg_local", "hops": hops})

        if bool(record.get("implicit_mechanism_edges", False)):
            for emit_id, emit_keys in emit_stmt_ids:
                for state_id, state_keys in state_landing_stmt_ids:
                    if emit_id == state_id:
                        continue
                    shared = sorted(emit_keys.intersection(state_keys))
                    if not shared and not emit_keys:
                        continue
                    add_edge(
                        g,
                        emit_id,
                        state_id,
                        "implicit_mechanism",
                        {
                            "kind": "event_state_link",
                            "confidence": 0.90 if shared else 0.72,
                            "evidence": "|".join(shared) if shared else "emit_state_sequence",
                        },
                    )
                    implicit_mechanism_edges_total += 1
            for verify_id, verify_keys in verify_stmt_ids:
                for exec_id, exec_keys in execute_stmt_ids:
                    if verify_id == exec_id:
                        continue
                    shared = sorted(verify_keys.intersection(exec_keys))
                    if not shared and not (verify_keys or exec_keys):
                        continue
                    add_edge(
                        g,
                        verify_id,
                        exec_id,
                        "implicit_mechanism",
                        {
                            "kind": "verify_execute_link",
                            "confidence": 0.92 if shared else 0.76,
                            "evidence": "|".join(shared) if shared else "verify_execute_sequence",
                        },
                    )
                    implicit_mechanism_edges_total += 1
            for emit_id, emit_keys in emit_stmt_ids:
                if not emit_keys:
                    continue
                for verify_id, verify_keys in verify_stmt_ids:
                    if emit_id == verify_id:
                        continue
                    shared = sorted(emit_keys.intersection(verify_keys))
                    if not shared:
                        continue
                    add_edge(
                        g,
                        emit_id,
                        verify_id,
                        "implicit_mechanism",
                        {
                            "kind": "emit_verify_candidate",
                            "confidence": 0.84,
                            "evidence": "|".join(shared),
                        },
                    )
                    implicit_mechanism_edges_total += 1

    function_summary_edges_total = 0
    function_state_summary_edges_total = 0
    interproc_state_flow_edges_total = 0
    cross_object_edges_total = 0
    interproc_function_ids: Set[int] = set()

    for summary in fn_summaries.values():
        fn_id = int(summary["fn_id"])
        for callee_fn_id in sorted(int(x) for x in summary["local_calls"]):
            if callee_fn_id == fn_id:
                continue
            add_edge(g, fn_id, callee_fn_id, "call_interaction", {"kind": "local_call_summary"})
            function_summary_edges_total += 1
            interproc_function_ids.add(fn_id)
            interproc_function_ids.add(callee_fn_id)

        for sv in sorted(str(x) for x in summary["state_write_vars"]):
            sv_id = state_var_nodes.get(sv)
            if sv_id is None:
                continue
            add_edge(g, fn_id, sv_id, "data_dep", {"kind": "state_write_summary", "var": sv})
            function_state_summary_edges_total += 1
        for sv in sorted(str(x) for x in summary["state_read_vars"]):
            sv_id = state_var_nodes.get(sv)
            if sv_id is None:
                continue
            add_edge(g, sv_id, fn_id, "data_dep", {"kind": "state_read_summary", "var": sv})
            function_state_summary_edges_total += 1

    for sv in sorted(state_var_nodes.keys()):
        writers: Dict[int, int] = {}
        readers: Dict[int, int] = {}
        for summary in fn_summaries.values():
            fn_id = int(summary["fn_id"])
            write_ids = pick_representative_ids(summary["write_stmt_by_var"].get(sv, []), limit=1)
            read_ids = pick_representative_ids(summary["read_stmt_by_var"].get(sv, []), limit=1)
            if write_ids:
                writers[fn_id] = int(write_ids[0])
            if read_ids:
                readers[fn_id] = int(read_ids[0])
        if not writers or not readers:
            continue
        pair_budget = MAX_INTERPROC_STATE_PAIRS_PER_VAR
        for writer_fn, writer_stmt in sorted(writers.items()):
            for reader_fn, reader_stmt in sorted(readers.items()):
                if writer_fn == reader_fn or writer_stmt == reader_stmt:
                    continue
                add_edge(
                    g,
                    writer_stmt,
                    reader_stmt,
                    "dfg_dep",
                    {"kind": "cross_function_state_flow", "var": sv},
                )
                interproc_state_flow_edges_total += 1
                interproc_function_ids.add(writer_fn)
                interproc_function_ids.add(reader_fn)
                pair_budget -= 1
                if pair_budget <= 0:
                    break
            if pair_budget <= 0:
                break

    for obj, obj_id in sorted(data_obj_nodes.items()):
        if obj not in INTERPROC_OBJECT_KEYS:
            continue
        rep_stmt_by_fn: Dict[int, int] = {}
        for summary in fn_summaries.values():
            stmt_ids = pick_representative_ids(summary["object_stmt_by_key"].get(obj, []), limit=1)
            if not stmt_ids:
                continue
            fn_id = int(summary["fn_id"])
            rep_stmt_by_fn[fn_id] = int(stmt_ids[0])
        if len(rep_stmt_by_fn) < 2:
            continue
        for fn_id, stmt_id in sorted(rep_stmt_by_fn.items()):
            add_edge(g, obj_id, stmt_id, "data_dep", {"kind": "shared_object_summary", "object": obj})
            cross_object_edges_total += 1
            interproc_function_ids.add(fn_id)

    g["edges"] = dedup_edges(g["edges"])
    node_type_cnt: Dict[str, int] = collections.Counter(str(n.get("type", "")) for n in g.get("nodes", []))
    edge_type_cnt: Dict[str, int] = collections.Counter(str(e.get("type", "")) for e in g.get("edges", []))

    ast_edge_count = int(sum(edge_type_cnt.get(t, 0) for t in AST_EDGE_TYPES))
    cfg_edge_count = int(sum(edge_type_cnt.get(t, 0) for t in CFG_EDGE_TYPES))
    dfg_edge_count = int(sum(edge_type_cnt.get(t, 0) for t in DFG_EDGE_TYPES))
    cfg_structured_edge_count = int(
        sum(
            1
            for e in g.get("edges", [])
            if e.get("type") == "cfg_next" and str(e.get("kind", "")) == "structured_cfg"
        )
    )
    cfg_sequential_edge_count = int(
        sum(
            1
            for e in g.get("edges", [])
            if e.get("type") == "cfg_next" and str(e.get("kind", "")) == "sequential"
        )
    )
    parser_ast_ok = bool(record.get("ast_parse_ok"))

    g["meta"]["node_count"] = len(g["nodes"])
    g["meta"]["edge_count"] = len(g["edges"])
    g["meta"]["function_count"] = sum(1 for n in g["nodes"] if n["type"] == "function")
    g["meta"]["node_type_count"] = dict(sorted(node_type_cnt.items()))
    g["meta"]["edge_type_count"] = dict(sorted(edge_type_cnt.items()))
    g["meta"]["ast_edge_count"] = ast_edge_count
    g["meta"]["cfg_edge_count"] = cfg_edge_count
    g["meta"]["dfg_edge_count"] = dfg_edge_count
    g["meta"]["cfg_structured_edge_count"] = cfg_structured_edge_count
    g["meta"]["cfg_sequential_edge_count"] = cfg_sequential_edge_count
    g["meta"]["function_summary_edges_total"] = int(function_summary_edges_total)
    g["meta"]["function_state_summary_edges_total"] = int(function_state_summary_edges_total)
    g["meta"]["interproc_state_flow_edges_total"] = int(interproc_state_flow_edges_total)
    g["meta"]["cross_object_edges_total"] = int(cross_object_edges_total)
    g["meta"]["implicit_mechanism_edges_total"] = int(implicit_mechanism_edges_total)
    g["meta"]["interproc_function_count"] = int(len(interproc_function_ids))
    g["meta"]["has_interproc_flow"] = bool(
        function_summary_edges_total > 0
        or function_state_summary_edges_total > 0
        or interproc_state_flow_edges_total > 0
        or cross_object_edges_total > 0
    )
    g["meta"]["has_implicit_mechanism"] = bool(implicit_mechanism_edges_total > 0)
    g["meta"]["has_ast"] = bool(ast_edge_count > 0)
    g["meta"]["has_cfg"] = bool(cfg_edge_count > 0)
    g["meta"]["has_dfg"] = bool(dfg_edge_count > 0)
    g["meta"]["has_cfg_structured"] = bool(cfg_structured_edge_count > 0)
    g["meta"]["has_ast_parse_backed"] = bool(parser_ast_ok and ast_edge_count > 0)
    g["meta"]["ast_cfg_dfg_complete"] = bool(
        g["meta"]["has_ast"] and g["meta"]["has_cfg"] and g["meta"]["has_dfg"]
    )
    return g


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1: build hetero program graphs from normalized contracts.")
    parser.add_argument("--input", type=str, default="data/processed/contracts_normalized.jsonl")
    parser.add_argument("--graph-dir", type=str, default="data/graphs/full")
    parser.add_argument("--index-out", type=str, default="data/graphs/graph_index.jsonl")
    parser.add_argument("--report-out", type=str, default="", help="Optional semantic coverage report json path.")
    parser.add_argument("--implicit-mechanism-edges", action="store_true", help="Inject high-confidence implicit mechanism edges such as emit->state and verify->execute.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = (ROOT / args.input).resolve()
    graph_dir = (ROOT / args.graph_dir).resolve()
    index_out = (ROOT / args.index_out).resolve()
    graph_dir.mkdir(parents=True, exist_ok=True)

    contracts = load_jsonl(in_path)
    index_rows: List[Dict[str, Any]] = []
    agg_node_type: Dict[str, int] = collections.Counter()
    agg_edge_type: Dict[str, int] = collections.Counter()
    graphs_with_ast = 0
    graphs_with_cfg = 0
    graphs_with_dfg = 0
    graphs_complete = 0
    graphs_ast_parse_ok = 0
    graphs_with_cfg_structured = 0
    graphs_ast_parse_backed = 0
    graphs_with_interproc_flow = 0
    graphs_with_implicit_mechanism = 0
    function_summary_edges_total = 0
    function_state_summary_edges_total = 0
    interproc_state_flow_edges_total = 0
    cross_object_edges_total = 0
    implicit_mechanism_edges_total = 0
    for row in contracts:
        row["implicit_mechanism_edges"] = bool(args.implicit_mechanism_edges)
        g = build_graph(row)
        out_path = graph_dir / f"{g['contract_id']}.json"
        save_json(out_path, g)
        for k, v in (g.get("meta", {}).get("node_type_count", {}) or {}).items():
            agg_node_type[str(k)] += int(v)
        for k, v in (g.get("meta", {}).get("edge_type_count", {}) or {}).items():
            agg_edge_type[str(k)] += int(v)
        graphs_with_ast += int(bool(g.get("meta", {}).get("has_ast", False)))
        graphs_with_cfg += int(bool(g.get("meta", {}).get("has_cfg", False)))
        graphs_with_dfg += int(bool(g.get("meta", {}).get("has_dfg", False)))
        graphs_complete += int(bool(g.get("meta", {}).get("ast_cfg_dfg_complete", False)))
        graphs_ast_parse_ok += int(bool(g.get("meta", {}).get("ast_parse_ok", False)))
        graphs_with_cfg_structured += int(bool(g.get("meta", {}).get("has_cfg_structured", False)))
        graphs_ast_parse_backed += int(bool(g.get("meta", {}).get("has_ast_parse_backed", False)))
        graphs_with_interproc_flow += int(bool(g.get("meta", {}).get("has_interproc_flow", False)))
        graphs_with_implicit_mechanism += int(bool(g.get("meta", {}).get("has_implicit_mechanism", False)))
        function_summary_edges_total += int(g.get("meta", {}).get("function_summary_edges_total", 0))
        function_state_summary_edges_total += int(g.get("meta", {}).get("function_state_summary_edges_total", 0))
        interproc_state_flow_edges_total += int(g.get("meta", {}).get("interproc_state_flow_edges_total", 0))
        cross_object_edges_total += int(g.get("meta", {}).get("cross_object_edges_total", 0))
        implicit_mechanism_edges_total += int(g.get("meta", {}).get("implicit_mechanism_edges_total", 0))
        index_rows.append(
            {
                "contract_id": g["contract_id"],
                "source_path": g["source_path"],
                "relative_source_path": g["meta"].get("relative_source_path"),
                "graph_path": str(out_path),
                "node_count": g["meta"]["node_count"],
                "edge_count": g["meta"]["edge_count"],
                "ast_parse_ok": bool(g.get("meta", {}).get("ast_parse_ok", False)),
                "parser_backend": g.get("meta", {}).get("parser_backend"),
                "ast_edge_count": int(g.get("meta", {}).get("ast_edge_count", 0)),
                "cfg_edge_count": int(g.get("meta", {}).get("cfg_edge_count", 0)),
                "dfg_edge_count": int(g.get("meta", {}).get("dfg_edge_count", 0)),
                "cfg_structured_edge_count": int(g.get("meta", {}).get("cfg_structured_edge_count", 0)),
                "cfg_sequential_edge_count": int(g.get("meta", {}).get("cfg_sequential_edge_count", 0)),
                "has_ast": bool(g.get("meta", {}).get("has_ast", False)),
                "has_cfg": bool(g.get("meta", {}).get("has_cfg", False)),
                "has_dfg": bool(g.get("meta", {}).get("has_dfg", False)),
                "has_cfg_structured": bool(g.get("meta", {}).get("has_cfg_structured", False)),
                "has_ast_parse_backed": bool(g.get("meta", {}).get("has_ast_parse_backed", False)),
                "ast_cfg_dfg_complete": bool(g.get("meta", {}).get("ast_cfg_dfg_complete", False)),
                "has_interproc_flow": bool(g.get("meta", {}).get("has_interproc_flow", False)),
                "function_summary_edges_total": int(g.get("meta", {}).get("function_summary_edges_total", 0)),
                "function_state_summary_edges_total": int(g.get("meta", {}).get("function_state_summary_edges_total", 0)),
                "interproc_state_flow_edges_total": int(g.get("meta", {}).get("interproc_state_flow_edges_total", 0)),
                "cross_object_edges_total": int(g.get("meta", {}).get("cross_object_edges_total", 0)),
                "implicit_mechanism_edges_total": int(g.get("meta", {}).get("implicit_mechanism_edges_total", 0)),
            }
        )
    save_jsonl(index_out, index_rows)
    if args.report_out:
        report_out = (ROOT / args.report_out).resolve()
        n_graphs = max(1, len(index_rows))
        report = {
            "graphs_total": int(len(index_rows)),
            "graphs_with_ast": int(graphs_with_ast),
            "graphs_with_cfg": int(graphs_with_cfg),
            "graphs_with_dfg": int(graphs_with_dfg),
            "graphs_ast_parse_ok": int(graphs_ast_parse_ok),
            "graphs_with_cfg_structured": int(graphs_with_cfg_structured),
            "graphs_with_ast_parse_backed": int(graphs_ast_parse_backed),
            "graphs_complete_ast_cfg_dfg": int(graphs_complete),
            "ratio_with_ast": float(graphs_with_ast / n_graphs),
            "ratio_with_cfg": float(graphs_with_cfg / n_graphs),
            "ratio_with_dfg": float(graphs_with_dfg / n_graphs),
            "ratio_ast_parse_ok": float(graphs_ast_parse_ok / n_graphs),
            "ratio_with_cfg_structured": float(graphs_with_cfg_structured / n_graphs),
            "ratio_with_ast_parse_backed": float(graphs_ast_parse_backed / n_graphs),
            "ratio_complete_ast_cfg_dfg": float(graphs_complete / n_graphs),
            "graphs_with_interproc_flow": int(graphs_with_interproc_flow),
            "graphs_with_implicit_mechanism": int(graphs_with_implicit_mechanism),
            "ratio_with_interproc_flow": float(graphs_with_interproc_flow / n_graphs),
            "ratio_with_implicit_mechanism": float(graphs_with_implicit_mechanism / n_graphs),
            "function_summary_edges_total": int(function_summary_edges_total),
            "function_state_summary_edges_total": int(function_state_summary_edges_total),
            "interproc_state_flow_edges_total": int(interproc_state_flow_edges_total),
            "cross_object_edges_total": int(cross_object_edges_total),
            "implicit_mechanism_edges_total": int(implicit_mechanism_edges_total),
            "aggregate_node_type_count": dict(sorted(agg_node_type.items())),
            "aggregate_edge_type_count": dict(sorted(agg_edge_type.items())),
        }
        save_json(report_out, report)
    print(f"[build_hetero_graph] graphs={len(index_rows)} dir={graph_dir}")


if __name__ == "__main__":
    main()
