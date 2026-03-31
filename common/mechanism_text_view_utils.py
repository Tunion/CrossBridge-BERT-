from __future__ import annotations

import collections
import re
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


KEY_REL = {
    "control_dep",
    "cfg_next",
    "data_dep",
    "dfg_dep",
    "call_interaction",
    "constraint_on",
    "implicit_mechanism",
    "contains",
    "ast_parent",
}
ROLE_SET = {"state_change", "external_interaction", "auth_constraint"}
ROLE_TRIPLETS: Sequence[Tuple[str, str]] = (
    ("auth_constraint", "external_interaction"),
    ("external_interaction", "state_change"),
    ("auth_constraint", "state_change"),
)
MESSAGE_KEYWORDS = (
    "message", "nonce", "payload", "proof", "signature", "root", "event", "relay",
    "bridge", "mint", "burn", "lock", "unlock", "claim", "release", "execute",
    "approve", "verify", "router", "endpoint", "gateway",
)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")


def _build_adj(graph: Dict[str, Any]) -> Dict[int, Set[int]]:
    adj: Dict[int, Set[int]] = collections.defaultdict(set)
    for e in graph.get("edges", []):
        if e.get("type") not in KEY_REL:
            continue
        s = int(e["src"])
        d = int(e["dst"])
        adj[s].add(d)
        adj[d].add(s)
    return adj


def _role_map(nodes: Iterable[Dict[str, Any]]) -> Dict[str, Set[int]]:
    out = {r: set() for r in ROLE_SET}
    for node in nodes:
        nid = int(node["id"])
        for role in node.get("roles", []) or []:
            if role in out:
                out[role].add(nid)
    return out


def _shortest_path_between_sets(
    adj: Dict[int, Set[int]],
    src_set: Set[int],
    dst_set: Set[int],
    max_hops: int,
) -> Optional[List[int]]:
    if not src_set or not dst_set:
        return None
    inter = src_set.intersection(dst_set)
    if inter:
        return [next(iter(inter))]
    q = collections.deque()
    parent: Dict[int, Optional[int]] = {}
    dist: Dict[int, int] = {}
    for s in src_set:
        q.append(s)
        parent[s] = None
        dist[s] = 0
    while q:
        cur = q.popleft()
        d = dist[cur]
        if d >= max_hops:
            continue
        for nb in adj.get(cur, set()):
            if nb in dist:
                continue
            dist[nb] = d + 1
            parent[nb] = cur
            if nb in dst_set:
                path = [nb]
                p = cur
                while p is not None:
                    path.append(p)
                    p = parent[p]
                path.reverse()
                return path
            q.append(nb)
    return None


def _clean_text(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"\s+", " ", raw)
    return raw[:180]


def _node_label(node: Dict[str, Any]) -> str:
    text = _clean_text(node.get("text") or node.get("name"))
    if text:
        return text
    node_type = str(node.get("type", "") or "").strip().lower()
    return node_type or "<node>"


def _uniq_keep_order(items: Iterable[str], limit: int) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        text = _clean_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_mechanism_text_view(
    slice_graph: Dict[str, Any],
    function_names: Iterable[str],
    seed_roles: Iterable[str],
    mechanism_path_max_len: int = 4,
) -> Tuple[str, Dict[str, Any]]:
    nodes = list(slice_graph.get("nodes", []))
    node_by_id = {int(n["id"]): n for n in nodes}
    adj = _build_adj(slice_graph)
    rmap = _role_map(nodes)

    role_snippets: Dict[str, List[str]] = {r: [] for r in ROLE_SET}
    message_snippets: List[str] = []
    for node in nodes:
        nid = int(node["id"])
        label = _node_label(node)
        lower = label.lower()
        for role in node.get("roles", []) or []:
            if role in role_snippets:
                role_snippets[role].append(label)
        if any(kw in lower for kw in MESSAGE_KEYWORDS):
            role_hits = ",".join(sorted(node.get("roles", []) or []))
            prefix = f"{node.get('type','node')}#{nid}"
            if role_hits:
                prefix += f"[{role_hits}]"
            message_snippets.append(f"{prefix}: {label}")

    path_sections: List[str] = []
    mechanism_links = 0
    for a, b in ROLE_TRIPLETS:
        path = _shortest_path_between_sets(adj, rmap.get(a, set()), rmap.get(b, set()), max_hops=mechanism_path_max_len)
        if not path:
            continue
        mechanism_links += 1
        labels = [_node_label(node_by_id.get(nid, {})) for nid in path]
        labels = [x for x in labels if x]
        if labels:
            path_sections.append(f"{a}->{b}: " + " -> ".join(labels[:8]))

    functions_text = " | ".join(sorted({str(x).strip().lower() for x in function_names if str(x).strip()}))
    seeds_text = " | ".join(sorted({str(x).strip().lower() for x in seed_roles if str(x).strip()}))

    lines: List[str] = []
    if functions_text:
        lines.append(f"[FUNCTIONS] {functions_text}")
    if seeds_text:
        lines.append(f"[SEED_ROLES] {seeds_text}")
    auth_items = _uniq_keep_order(role_snippets["auth_constraint"], limit=6)
    ext_items = _uniq_keep_order(role_snippets["external_interaction"], limit=6)
    state_items = _uniq_keep_order(role_snippets["state_change"], limit=6)
    msg_items = _uniq_keep_order(message_snippets, limit=8)
    if auth_items:
        lines.append("[AUTH] " + " || ".join(auth_items))
    if ext_items:
        lines.append("[EXTERNAL] " + " || ".join(ext_items))
    if state_items:
        lines.append("[STATE] " + " || ".join(state_items))
    if msg_items:
        lines.append("[MESSAGE] " + " || ".join(msg_items))
    if path_sections:
        lines.append("[PATHS] " + " || ".join(path_sections))
    if not lines:
        lines.append("[EMPTY] no salient mechanism text")
    text = "\n".join(lines)
    meta = {
        "mechanism_text_len": int(len(text)),
        "mechanism_text_has_auth": bool(auth_items),
        "mechanism_text_has_external": bool(ext_items),
        "mechanism_text_has_state": bool(state_items),
        "mechanism_text_message_hits": int(len(msg_items)),
        "mechanism_text_paths": int(mechanism_links),
    }
    return text, meta


def tokenize_mechanism_text(text: Any, vocab_size: int = 8192, max_tokens: int = 96) -> List[int]:
    raw = str(text or "").strip().lower()
    if not raw:
        return [0]
    toks = TOKEN_RE.findall(raw)
    if not toks:
        return [0]
    limit = max(1, int(max_tokens))
    denom = max(1, int(vocab_size) - 1)
    out: List[int] = []
    for tok in toks[:limit]:
        hid = (zlib.crc32(tok.encode("utf-8")) % denom) + 1
        out.append(int(hid))
    return out or [0]
