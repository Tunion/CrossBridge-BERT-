from __future__ import annotations

import argparse
import io
import re
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.graph_utils import make_contract_id
from common.io_utils import resolve_solidity_paths, save_jsonl, to_posix

try:
    from solidity_parser import parser as solidity_parser
except Exception:  # pragma: no cover
    solidity_parser = None


FUNCTION_DEF_RE = re.compile(r"^\s*function\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\(")
CONSTRUCTOR_RE = re.compile(r"^\s*constructor\s*\(")
FALLBACK_RE = re.compile(r"^\s*fallback\s*\(")
RECEIVE_RE = re.compile(r"^\s*receive\s*\(")
MODIFIER_DEF_RE = re.compile(r"^\s*modifier\s+([A-Za-z_][A-Za-z0-9_]*)\b")

CONSTRAINT_PATTERNS = [
    r"\brequire\s*\(",
    r"\bassert\s*\(",
    r"\brevert\s*\(",
    r"\bonlyowner\b",
    r"\bonlyadmin\b",
    r"\bmsg\.sender\b",
    r"\bsignature\b",
    r"\bproof\b",
    r"\becrecover\b",
    r"\bverify\b",
    r"\bnonce\b",
]
EXTERNAL_PATTERNS = [
    r"\.call\s*\(",
    r"\.delegatecall\s*\(",
    r"\.staticcall\s*\(",
    r"\.send\s*\(",
    r"\.transfer\s*\(",
    r"\brouter\b",
    r"\bbridge\b",
    r"\binvoke\b",
    r"\bexecute\b",
    r"\bexternal\b",
]
STATE_PATTERNS = [
    r"[A-Za-z_][A-Za-z0-9_\.\[\]]*\s*[\+\-\*\/]?=",
    r"\+\+",
    r"--",
    r"\bmint\b",
    r"\bburn\b",
    r"\block\b",
    r"\bunlock\b",
    r"\bbalance\b",
    r"\bmapping\b",
]

SEMANTIC_PRIORITY = ["constraint_check", "external_interaction", "state_change", "context"]
DATA_OBJECT_KEYS = {
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
}
ASSIGNMENT_OPS = {"=", "+=", "-=", "*=", "/=", "%=", "|=", "&=", "^=", "<<=", ">>="}
INC_DEC_OPS = {"++", "--"}
LOW_LEVEL_CALL_MEMBERS = {"call", "delegatecall", "staticcall", "send", "transfer"}
CONSTRAINT_CALLS = {"require", "assert", "revert"}
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


def _safe_dict(node: Any) -> Dict[str, Any]:
    return node if isinstance(node, dict) else {}


def _node_type(node: Any) -> str:
    return _safe_dict(node).get("type", "")


def _sort_semantic_tags(tags: Iterable[str]) -> List[str]:
    values = set(tags)
    if not values:
        return ["context"]
    out = [x for x in SEMANTIC_PRIORITY if x in values]
    return out if out else ["context"]


def detect_semantic_classes(text: str) -> List[str]:
    low = text.lower()
    out: List[str] = []
    if any(re.search(p, low) for p in CONSTRAINT_PATTERNS):
        out.append("constraint_check")
    if any(re.search(p, low) for p in EXTERNAL_PATTERNS):
        out.append("external_interaction")
    if any(re.search(p, low) for p in STATE_PATTERNS):
        out.append("state_change")
    if not out:
        out.append("context")
    return _sort_semantic_tags(out)


def classify_statement(line: str) -> Tuple[str, List[str], List[str]]:
    line_strip = line.strip()
    syntax_tags: List[str] = []
    stmt_type = "statement"
    if line_strip.startswith("//") or line_strip.startswith("/*") or line_strip.startswith("*"):
        stmt_type = "comment"
        syntax_tags.append("comment")
    elif line_strip.startswith("if ") or line_strip.startswith("if("):
        stmt_type = "if"
    elif line_strip.startswith("for ") or line_strip.startswith("for("):
        stmt_type = "loop"
    elif line_strip.startswith("while ") or line_strip.startswith("while("):
        stmt_type = "loop"
    elif line_strip.startswith("emit "):
        stmt_type = "emit"
    elif "require(" in line_strip or "assert(" in line_strip or "revert(" in line_strip:
        stmt_type = "constraint"
    elif ".call(" in line_strip or ".delegatecall(" in line_strip or ".transfer(" in line_strip:
        stmt_type = "call"
    elif "=" in line_strip:
        stmt_type = "assign"
    semantic = detect_semantic_classes(line_strip)
    return stmt_type, semantic, syntax_tags


def _collect_header(lines: List[str], start_idx: int) -> Tuple[str, int]:
    if start_idx < 1 or start_idx > len(lines):
        return "", start_idx
    buf = [lines[start_idx - 1]]
    i = start_idx
    while i < len(lines):
        if "{" in buf[-1] or ";" in buf[-1]:
            break
        i += 1
        if i > len(lines):
            break
        buf.append(lines[i - 1])
    return " ".join(s.strip() for s in buf), i


def _is_comment_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("//") or s.startswith("/*") or s.startswith("*") or s.startswith("*/")


def collect_leading_comment_block(lines: List[str], start_idx: int, max_lookback: int = 8) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    j = start_idx - 1
    steps = 0
    while j >= 1 and steps < max_lookback:
        raw = lines[j - 1]
        s = raw.strip()
        if s == "":
            if out:
                break
            j -= 1
            steps += 1
            continue
        if _is_comment_line(raw):
            out.append((j, raw))
            j -= 1
            steps += 1
            continue
        break
    out.reverse()
    return out


def parse_inline_statements(header: str, line_no: int) -> List[Tuple[int, str]]:
    if "{" not in header or "}" not in header:
        return []
    body = header.split("{", 1)[1].rsplit("}", 1)[0].strip()
    if not body:
        return []
    return [(line_no, f"{x.strip()};") for x in body.split(";") if x.strip()]


def parse_function_name(line: str) -> str:
    m = FUNCTION_DEF_RE.match(line)
    if m:
        return m.group(1) or "<anonymous_function>"
    if CONSTRUCTOR_RE.match(line):
        return "constructor"
    if RECEIVE_RE.match(line):
        return "receive"
    if FALLBACK_RE.match(line):
        return "fallback"
    return "unknown"


def parse_function_modifiers(header: str, known_modifiers: Dict[str, Dict[str, Any]]) -> List[str]:
    after = header.split(")", 1)[1] if ")" in header else header
    tokens = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", after)
    blocked = {
        "public",
        "private",
        "internal",
        "external",
        "view",
        "pure",
        "payable",
        "virtual",
        "override",
        "returns",
        "memory",
        "calldata",
        "storage",
        "constant",
    }
    return [t for t in tokens if t not in blocked and t in known_modifiers]


def extract_modifier_defs_heuristic(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    mod_map: Dict[str, Dict[str, Any]] = {}
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        m = MODIFIER_DEF_RE.match(line)
        if not m:
            i += 1
            continue
        name = m.group(1)
        header, end_header_line = _collect_header(lines, i)
        depth = header.count("{") - header.count("}")
        body_lines: List[Tuple[int, str]] = []
        j = end_header_line + 1
        while depth > 0 and j <= len(lines):
            cur = lines[j - 1]
            body_lines.append((j, cur))
            depth += cur.count("{") - cur.count("}")
            j += 1
        sem = sorted({c for _, text in body_lines for c in detect_semantic_classes(text) if c != "context"})
        mod_map[name] = {
            "name": name,
            "start_line": i,
            "end_line": max(i, j - 1),
            "header": header,
            "semantic_tags": _sort_semantic_tags(sem if sem else ["constraint_check"]),
        }
        i = max(i + 1, j)
    return mod_map


def _mk_stmt(idx: int, line: Optional[int], text: str, stmt_type: str, semantic: Sequence[str], syntax_tags: Sequence[str]) -> Dict[str, Any]:
    return {
        "index": idx,
        "line": line,
        "end_line": line,
        "text": text.strip(),
        "stmt_type": stmt_type,
        "semantic_class": _sort_semantic_tags(semantic),
        "syntax_tags": list(syntax_tags),
        "ast_type": None,
        "parent_stmt_index": None,
        "cfg_succ_indices": [],
        "defs": [],
        "uses": [],
        "state_defs": [],
        "data_objects": sorted({x for x in DATA_OBJECT_KEYS if x in text.lower()}),
        "call_targets": [],
    }


def extract_functions_heuristic(
    lines: List[str],
    modifiers: Dict[str, Dict[str, Any]],
    expand_modifiers: bool,
    keep_comments: bool,
) -> List[Dict[str, Any]]:
    functions: List[Dict[str, Any]] = []
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        if not (FUNCTION_DEF_RE.match(line) or CONSTRUCTOR_RE.match(line) or FALLBACK_RE.match(line) or RECEIVE_RE.match(line)):
            i += 1
            continue
        name = parse_function_name(line)
        header, end_header_line = _collect_header(lines, i)
        fn_mods = parse_function_modifiers(header, modifiers)
        depth = header.count("{") - header.count("}")
        body: List[Tuple[int, str]] = []
        if depth <= 0:
            body.extend(parse_inline_statements(header, end_header_line))
        j = end_header_line + 1
        while depth > 0 and j <= len(lines):
            cur = lines[j - 1]
            body.append((j, cur))
            depth += cur.count("{") - cur.count("}")
            j += 1

        statements: List[Dict[str, Any]] = []
        if keep_comments:
            for ln, text in collect_leading_comment_block(lines, i):
                st, sem, tags = classify_statement(text)
                statements.append(_mk_stmt(len(statements), ln, text, st, sem, tags))
        if expand_modifiers and fn_mods:
            for mod in fn_mods:
                sem = modifiers[mod].get("semantic_tags", ["constraint_check"])
                statements.append(_mk_stmt(len(statements), i, f"[expanded_modifier] {mod}", "expanded_modifier", sem, ["modifier_expansion"]))
        for ln, text in body:
            st, sem, tags = classify_statement(text)
            if st == "comment" and not keep_comments:
                continue
            statements.append(_mk_stmt(len(statements), ln, text, st, sem, tags))
        for k in range(len(statements) - 1):
            statements[k]["cfg_succ_indices"] = [k + 1]
        functions.append(
            {
                "name": name,
                "contract_name": None,
                "start_line": i,
                "end_line": max(i, j - 1),
                "header": header,
                "modifiers": fn_mods,
                "statements": statements,
            }
        )
        i = max(i + 1, j)
    return functions


def parse_solidity_ast_safe(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if solidity_parser is None:
        return None, "solidity_parser_not_installed"
    try:
        out = io.StringIO()
        err = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            ast = solidity_parser.parse(text, loc=True)
        return _safe_dict(ast), None
    except Exception as e:  # pragma: no cover
        return None, str(e)


def _iter_ast_nodes(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for v in node.values():
            if isinstance(v, (dict, list)):
                yield from _iter_ast_nodes(v)
    elif isinstance(node, list):
        for x in node:
            if isinstance(x, (dict, list)):
                yield from _iter_ast_nodes(x)


def _loc_lines(node: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    loc = _safe_dict(node).get("loc")
    if not isinstance(loc, dict):
        return None, None
    s = _safe_dict(loc.get("start")).get("line")
    e = _safe_dict(loc.get("end")).get("line")
    return (int(s) if isinstance(s, int) else None, int(e) if isinstance(e, int) else None)


def _extract_text_by_lines(lines: List[str], start_line: Optional[int], end_line: Optional[int], max_chars: int = 1200) -> str:
    if not start_line or start_line < 1:
        return ""
    end_line = end_line if end_line and end_line >= start_line else start_line
    s = max(1, start_line)
    e = min(len(lines), end_line)
    if s > len(lines) or s > e:
        return ""
    text = "\n".join(lines[s - 1 : e]).strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _collect_identifiers(node: Any) -> Set[str]:
    out: Set[str] = set()
    for n in _iter_ast_nodes(node):
        t = _node_type(n)
        if t == "Identifier":
            name = n.get("name")
            if isinstance(name, str) and name:
                out.add(name)
        elif t == "MemberAccess":
            member = n.get("memberName")
            if isinstance(member, str) and member:
                out.add(member)
    return out


def _extract_lvalue_names(node: Any) -> Set[str]:
    n = _safe_dict(node)
    t = n.get("type")
    if t == "Identifier":
        name = n.get("name")
        return {name} if isinstance(name, str) and name else set()
    if t == "MemberAccess":
        base = _extract_lvalue_names(n.get("expression"))
        member = n.get("memberName")
        if isinstance(member, str) and member:
            base.add(member)
        return base
    if t == "IndexAccess":
        return _extract_lvalue_names(n.get("base"))
    if t == "TupleExpression":
        out: Set[str] = set()
        for comp in n.get("components", []) or []:
            out.update(_extract_lvalue_names(comp))
        return out
    return set()


def _collect_defs_from_expression(expr: Any) -> Set[str]:
    e = _safe_dict(expr)
    if not e:
        return set()
    t = e.get("type")
    out: Set[str] = set()
    if t == "BinaryOperation" and e.get("operator") in ASSIGNMENT_OPS:
        out.update(_extract_lvalue_names(e.get("left")))
    elif t == "UnaryOperation" and e.get("operator") in INC_DEC_OPS:
        out.update(_extract_lvalue_names(e.get("subExpression")))
    return {x for x in out if x and x.lower() not in SOLIDITY_KEYWORDS}


def _extract_defs_uses_from_stmt(stmt_node: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    defs: Set[str] = set()
    t = _node_type(stmt_node)
    ident_scopes: List[Any] = []
    if t == "VariableDeclarationStatement":
        for decl in stmt_node.get("variables", []) or []:
            name = _safe_dict(decl).get("name")
            if isinstance(name, str) and name:
                defs.add(name)
        defs.update(_collect_defs_from_expression(stmt_node.get("initialValue")))
        ident_scopes.append(stmt_node.get("initialValue"))
    elif t == "ExpressionStatement":
        defs.update(_collect_defs_from_expression(stmt_node.get("expression")))
        ident_scopes.append(stmt_node.get("expression"))
    elif t == "ForStatement":
        defs.update(_collect_defs_from_expression(stmt_node.get("initExpression")))
        defs.update(_collect_defs_from_expression(stmt_node.get("loopExpression")))
        ident_scopes.extend([stmt_node.get("initExpression"), stmt_node.get("conditionExpression"), stmt_node.get("loopExpression")])
    elif t in {"IfStatement", "WhileStatement", "DoWhileStatement"}:
        ident_scopes.append(stmt_node.get("condition"))
    else:
        ident_scopes.append(stmt_node)

    idents: Set[str] = set()
    for scope in ident_scopes:
        idents.update(_collect_identifiers(scope))
    defs = {x for x in defs if x and x.lower() not in SOLIDITY_KEYWORDS}
    uses = {x for x in idents if x and x.lower() not in SOLIDITY_KEYWORDS and x not in defs}
    return sorted(defs), sorted(uses)


def _collect_call_targets(stmt_node: Dict[str, Any]) -> List[str]:
    out: Set[str] = set()
    for n in _iter_ast_nodes(stmt_node):
        if _node_type(n) != "FunctionCall":
            continue
        expr = _safe_dict(n.get("expression"))
        et = expr.get("type")
        if et == "Identifier":
            name = expr.get("name")
            if isinstance(name, str) and name:
                out.add(name)
        elif et == "MemberAccess":
            member = expr.get("memberName")
            base_name = ""
            base = _safe_dict(expr.get("expression"))
            if base.get("type") == "Identifier":
                bname = base.get("name")
                if isinstance(bname, str):
                    base_name = bname
            if isinstance(member, str) and member:
                out.add(f"{base_name}.{member}" if base_name else member)
        elif et:
            out.add(et)
    return sorted(out)


def _is_external_call_target(target: str) -> bool:
    low = target.lower()
    if any(low.endswith(f".{m}") or low == m for m in LOW_LEVEL_CALL_MEMBERS):
        return True
    return any(k in low for k in ("bridge", "router", "invoke", "execute"))


def _extract_data_objects_from_semantics(text: str, defs: Sequence[str], uses: Sequence[str]) -> List[str]:
    low = text.lower()
    out = {k for k in DATA_OBJECT_KEYS if k in low}
    for x in list(defs) + list(uses):
        lx = x.lower()
        if lx in DATA_OBJECT_KEYS:
            out.add(lx)
    return sorted(out)


def _classify_ast_statement(
    stmt_node: Dict[str, Any],
    text: str,
    defs: Sequence[str],
    uses: Sequence[str],
    state_defs: Sequence[str],
    call_targets: Sequence[str],
) -> Tuple[str, List[str], List[str]]:
    ast_type = _node_type(stmt_node)
    syntax_tags: List[str] = []
    stmt_type = "statement"
    if ast_type == "IfStatement":
        stmt_type = "if"
    elif ast_type in {"ForStatement", "WhileStatement", "DoWhileStatement"}:
        stmt_type = "loop"
    elif ast_type == "EmitStatement":
        stmt_type = "emit"
    elif ast_type == "ReturnStatement":
        stmt_type = "return"
    elif ast_type == "VariableDeclarationStatement":
        stmt_type = "var_decl"
    elif ast_type == "ExpressionStatement":
        expr = _safe_dict(stmt_node.get("expression"))
        et = expr.get("type")
        if et == "BinaryOperation" and expr.get("operator") in ASSIGNMENT_OPS:
            stmt_type = "assign"
        elif et == "UnaryOperation" and expr.get("operator") in INC_DEC_OPS:
            stmt_type = "assign"
        elif et == "FunctionCall":
            stmt_type = "call"
    elif ast_type in {"ThrowStatement"}:
        stmt_type = "constraint"
    elif ast_type == "UncheckedStatement":
        stmt_type = "unchecked"
    elif ast_type == "InLineAssemblyStatement":
        stmt_type = "assembly"

    sem: Set[str] = set(detect_semantic_classes(text))
    if "context" in sem and len(sem) > 1:
        sem.remove("context")

    call_l = [x.lower() for x in call_targets]
    if any(t in CONSTRAINT_CALLS or t.endswith(".require") or t.endswith(".assert") or t.endswith(".revert") for t in call_l):
        sem.add("constraint_check")
    if stmt_type in {"if", "constraint"}:
        sem.add("constraint_check")
    if any(_is_external_call_target(t) for t in call_targets):
        sem.add("external_interaction")
    if state_defs:
        sem.add("state_change")
    elif stmt_type in {"assign", "var_decl"} and defs and any(x not in uses for x in defs):
        sem.add("state_change")

    if not sem:
        sem.add("context")
    return stmt_type, _sort_semantic_tags(sem), syntax_tags


def _dedup_ordered_int(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _build_statements_from_ast_function(
    fn_node: Dict[str, Any],
    lines: List[str],
    local_vars: Set[str],
    state_vars: Set[str],
) -> List[Dict[str, Any]]:
    statements: List[Dict[str, Any]] = []
    cfg_edges: Set[Tuple[int, int]] = set()

    def add_cfg(src: Optional[int], dst: Optional[int]) -> None:
        if src is None or dst is None or src == dst:
            return
        if src < 0 or dst < 0 or src >= len(statements) or dst >= len(statements):
            return
        cfg_edges.add((src, dst))

    def add_stmt(node: Dict[str, Any], parent_idx: Optional[int]) -> int:
        idx = len(statements)
        s_line, e_line = _loc_lines(node)
        text = _extract_text_by_lines(lines, s_line, e_line)
        defs, uses = _extract_defs_uses_from_stmt(node)
        call_targets = _collect_call_targets(node)
        state_defs = sorted({x for x in defs if x in state_vars or x not in local_vars})
        st, sem, tags = _classify_ast_statement(node, text, defs, uses, state_defs, call_targets)
        statements.append(
            {
                "index": idx,
                "line": s_line,
                "end_line": e_line if e_line is not None else s_line,
                "text": text.strip(),
                "stmt_type": st,
                "semantic_class": sem,
                "syntax_tags": tags,
                "ast_type": _node_type(node),
                "parent_stmt_index": parent_idx,
                "cfg_succ_indices": [],
                "defs": defs,
                "uses": uses,
                "state_defs": state_defs,
                "data_objects": _extract_data_objects_from_semantics(text, defs, uses),
                "call_targets": call_targets,
            }
        )
        return idx

    def walk_list(stmt_list: Sequence[Any], parent_idx: Optional[int]) -> Tuple[Optional[int], List[int]]:
        entry: Optional[int] = None
        prev_exits: List[int] = []
        for st in stmt_list:
            s_entry, s_exits = walk_stmt(st, parent_idx)
            if s_entry is None:
                continue
            if entry is None:
                entry = s_entry
            for pe in prev_exits:
                add_cfg(pe, s_entry)
            prev_exits = s_exits
        return entry, prev_exits

    def walk_stmt(stmt_node: Any, parent_idx: Optional[int]) -> Tuple[Optional[int], List[int]]:
        node = _safe_dict(stmt_node)
        if not node:
            return None, []
        t = node.get("type")
        if t == "Block":
            return walk_list(node.get("statements", []) or [], parent_idx)

        cur = add_stmt(node, parent_idx)
        if t == "IfStatement":
            t_entry, t_exits = walk_stmt(node.get("TrueBody"), cur)
            f_entry, f_exits = walk_stmt(node.get("FalseBody"), cur)
            if t_entry is not None:
                add_cfg(cur, t_entry)
            if f_entry is not None:
                add_cfg(cur, f_entry)
            exits: List[int] = []
            exits.extend(t_exits if t_entry is not None else [cur])
            exits.extend(f_exits if f_entry is not None else [cur])
            return cur, _dedup_ordered_int(exits if exits else [cur])

        if t in {"ForStatement", "WhileStatement", "DoWhileStatement", "UncheckedStatement"}:
            b_entry, b_exits = walk_stmt(node.get("body"), cur)
            if b_entry is not None:
                add_cfg(cur, b_entry)
                for ex in b_exits:
                    add_cfg(ex, cur)
            return cur, [cur]

        if t == "TryStatement":
            exits: List[int] = []
            b_entry, b_exits = walk_stmt(node.get("body"), cur)
            if b_entry is not None:
                add_cfg(cur, b_entry)
                exits.extend(b_exits if b_exits else [cur])
            for clause in node.get("catchClauses", []) or []:
                c_body = _safe_dict(clause).get("body")
                c_entry, c_exits = walk_stmt(c_body, cur)
                if c_entry is not None:
                    add_cfg(cur, c_entry)
                    exits.extend(c_exits if c_exits else [c_entry])
            return cur, _dedup_ordered_int(exits if exits else [cur])

        return cur, [cur]

    body = _safe_dict(fn_node.get("body"))
    if body:
        walk_stmt(body, None)

    for src, dst in sorted(cfg_edges):
        succ = list(statements[src].get("cfg_succ_indices", []))
        succ.append(dst)
        statements[src]["cfg_succ_indices"] = _dedup_ordered_int([int(x) for x in succ if isinstance(x, int)])
    return statements


def _normalize_function_name(fn_node: Dict[str, Any]) -> str:
    if fn_node.get("isConstructor"):
        return "constructor"
    if fn_node.get("isReceive"):
        return "receive"
    if fn_node.get("isFallback"):
        return "fallback"
    name = fn_node.get("name")
    return name if isinstance(name, str) and name else "<anonymous_function>"


def _parse_modifier_invocations(items: Any) -> List[str]:
    out: List[str] = []
    for inv in items or []:
        name = _safe_dict(inv).get("name")
        if isinstance(name, str) and name:
            out.append(name)
    return out


def _collect_local_variables(fn_node: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for key in ("parameters", "returnParameters"):
        plist = _safe_dict(fn_node.get(key))
        for p in plist.get("parameters", []) or []:
            name = _safe_dict(p).get("name")
            if isinstance(name, str) and name:
                out.add(name)
    for n in _iter_ast_nodes(_safe_dict(fn_node.get("body"))):
        if _node_type(n) == "VariableDeclaration":
            name = n.get("name")
            if isinstance(name, str) and name:
                out.add(name)
    return out


def _collect_state_variables(contract_node: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for sub in contract_node.get("subNodes", []) or []:
        sd = _safe_dict(sub)
        if sd.get("type") != "StateVariableDeclaration":
            continue
        for var in sd.get("variables", []) or []:
            name = _safe_dict(var).get("name")
            if isinstance(name, str) and name:
                out.add(name)
    return out


def _shift_statements(statements: Sequence[Dict[str, Any]], offset: int) -> List[Dict[str, Any]]:
    if offset == 0:
        return [dict(x) for x in statements]
    out: List[Dict[str, Any]] = []
    for s in statements:
        ns = dict(s)
        idx = s.get("index")
        ns["index"] = (int(idx) + offset) if isinstance(idx, int) else offset
        parent = s.get("parent_stmt_index")
        ns["parent_stmt_index"] = (int(parent) + offset) if isinstance(parent, int) else None
        ns["cfg_succ_indices"] = [int(x) + offset for x in s.get("cfg_succ_indices", []) if isinstance(x, int)]
        out.append(ns)
    return out


def _build_prelude(
    lines: List[str],
    fn_line: int,
    fn_mods: Sequence[str],
    modifier_defs: Dict[str, Dict[str, Any]],
    keep_comments: bool,
    expand_modifiers: bool,
) -> List[Dict[str, Any]]:
    prelude: List[Dict[str, Any]] = []
    if keep_comments:
        for ln, text in collect_leading_comment_block(lines, fn_line):
            st, sem, tags = classify_statement(text)
            prelude.append(_mk_stmt(len(prelude), ln, text, st, sem, tags))
    if expand_modifiers:
        for mod in fn_mods:
            sem = modifier_defs.get(mod, {}).get("semantic_tags", ["constraint_check"])
            x = _mk_stmt(len(prelude), fn_line, f"[expanded_modifier] {mod}", "expanded_modifier", sem, ["modifier_expansion"])
            x["ast_type"] = "ExpandedModifier"
            x["call_targets"] = [mod]
            prelude.append(x)
    for i in range(len(prelude) - 1):
        prelude[i]["cfg_succ_indices"] = [i + 1]
    return prelude


def _extract_contract_nodes(ast_root: Dict[str, Any]) -> List[Dict[str, Any]]:
    children = ast_root.get("children", []) or []
    top = [x for x in children if _safe_dict(x).get("type") == "ContractDefinition"]
    if top:
        return top
    all_nodes = [n for n in _iter_ast_nodes(ast_root) if n.get("type") == "ContractDefinition"]
    all_nodes.sort(key=lambda n: (_loc_lines(n)[0] or 10**9, n.get("name", "")))
    return all_nodes


def extract_from_ast(
    ast_root: Dict[str, Any],
    lines: List[str],
    keep_comments: bool,
    expand_modifiers: bool,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    modifier_defs: Dict[str, Dict[str, Any]] = {}
    functions: List[Dict[str, Any]] = []

    for contract in _extract_contract_nodes(ast_root):
        contract_name = contract.get("name")
        state_vars = _collect_state_variables(contract)
        sub_nodes = contract.get("subNodes", []) or []

        for sub in sub_nodes:
            mod = _safe_dict(sub)
            if mod.get("type") != "ModifierDefinition":
                continue
            name = mod.get("name")
            if not isinstance(name, str) or not name:
                continue
            s_line, e_line = _loc_lines(mod)
            if s_line is None:
                continue
            header, _ = _collect_header(lines, s_line)
            body_text = _extract_text_by_lines(lines, *_loc_lines(_safe_dict(mod.get("body")) or mod))
            sem = [x for x in detect_semantic_classes(body_text) if x != "context"]
            modifier_defs[name] = {
                "name": name,
                "contract_name": contract_name,
                "start_line": s_line,
                "end_line": e_line if e_line is not None else s_line,
                "header": header,
                "semantic_tags": _sort_semantic_tags(sem if sem else ["constraint_check"]),
            }

        for sub in sub_nodes:
            fn = _safe_dict(sub)
            if fn.get("type") != "FunctionDefinition":
                continue
            fn_name = _normalize_function_name(fn)
            s_line, e_line = _loc_lines(fn)
            if s_line is None:
                continue
            header, _ = _collect_header(lines, s_line)
            fn_mods = _parse_modifier_invocations(fn.get("modifiers"))
            local_vars = _collect_local_variables(fn)
            ast_stmts = _build_statements_from_ast_function(fn, lines, local_vars, state_vars)
            prelude = _build_prelude(lines, s_line, fn_mods, modifier_defs, keep_comments, expand_modifiers)
            merged = prelude + _shift_statements(ast_stmts, len(prelude))
            if prelude and ast_stmts:
                first_ast = merged[len(prelude)]["index"]
                prelude_tail = merged[len(prelude) - 1]
                prelude_tail["cfg_succ_indices"] = _dedup_ordered_int(
                    [int(x) for x in list(prelude_tail.get("cfg_succ_indices", [])) + [first_ast] if isinstance(x, int)]
                )
            functions.append(
                {
                    "name": fn_name,
                    "contract_name": contract_name,
                    "start_line": s_line,
                    "end_line": e_line if e_line is not None else s_line,
                    "header": header,
                    "modifiers": fn_mods,
                    "statements": merged,
                }
            )

    functions.sort(key=lambda f: (f.get("start_line") or 10**9, f.get("name") or ""))
    return modifier_defs, functions


def normalize_contract(
    project_root: Path,
    source_path: Path,
    keep_comments: bool,
    expand_modifiers: bool,
    max_lines: int,
) -> Dict[str, Any]:
    text = source_path.read_text(encoding="utf-8", errors="ignore")
    lines_all = text.splitlines()
    if max_lines and len(lines_all) > max_lines:
        lines = lines_all[:max_lines]
        text_for_parse = "\n".join(lines)
    else:
        lines = lines_all
        text_for_parse = text

    ast_root, ast_error = parse_solidity_ast_safe(text_for_parse)
    parser_backend = "heuristic_regex"
    if ast_root is not None:
        mods_map, functions = extract_from_ast(
            ast_root=ast_root,
            lines=lines,
            keep_comments=keep_comments,
            expand_modifiers=expand_modifiers,
        )
        modifiers = list(mods_map.values())
        parser_backend = "solidity_parser_ast"
    else:
        mods_map = extract_modifier_defs_heuristic(lines)
        functions = extract_functions_heuristic(lines, mods_map, expand_modifiers, keep_comments)
        modifiers = list(mods_map.values())

    rel_path = source_path.resolve().relative_to(project_root.resolve())
    comment_lines = sum(1 for x in lines if x.strip().startswith("//")) if keep_comments else 0
    return {
        "contract_id": make_contract_id(to_posix(rel_path)),
        "source_path": str(source_path.resolve()),
        "relative_source_path": to_posix(rel_path),
        "file_name": source_path.name,
        "line_count": len(lines),
        "comment_lines": comment_lines,
        "parser_backend": parser_backend,
        "ast_parse_ok": ast_root is not None,
        "ast_parse_error": ast_error,
        "modifiers": modifiers,
        "functions": functions,
        "summary": {
            "function_count": len(functions),
            "modifier_count": len(modifiers),
            "statement_count": sum(len(f["statements"]) for f in functions),
            "parser_backend": parser_backend,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1: structure expansion and semantic normalization.")
    parser.add_argument("--project-root", type=str, default=".", help="Project root path.")
    parser.add_argument("--dataset-root", type=str, default="DataSet", help="Dataset root directory.")
    parser.add_argument(
        "--input-list",
        type=str,
        default="processed/dataset_non_overlap_local.txt",
        help="Optional solidity path list.",
    )
    parser.add_argument("--output", type=str, default="data/processed/contracts_normalized.jsonl")
    parser.add_argument("--max-files", type=int, default=200, help="0 means all files.")
    parser.add_argument("--keep-comments", action="store_true", help="Keep comment statistics.")
    parser.add_argument("--expand-modifiers", action="store_true", help="Expand known modifier semantics.")
    parser.add_argument("--max-lines", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    dataset_root = (project_root / args.dataset_root).resolve()
    input_list = (project_root / args.input_list).resolve() if args.input_list else None
    output_path = (project_root / args.output).resolve()

    sol_paths = resolve_solidity_paths(
        project_root=project_root,
        dataset_root=dataset_root,
        input_list=input_list,
        max_files=args.max_files,
    )

    rows: List[Dict[str, Any]] = []
    ast_ok = 0
    for p in sol_paths:
        row = normalize_contract(
            project_root=project_root,
            source_path=p,
            keep_comments=bool(args.keep_comments),
            expand_modifiers=bool(args.expand_modifiers),
            max_lines=args.max_lines,
        )
        if row.get("ast_parse_ok"):
            ast_ok += 1
        rows.append(row)

    save_jsonl(output_path, rows)
    print(
        f"[normalize_or_expand] contracts={len(rows)} ast_ok={ast_ok} "
        f"ast_fallback={len(rows) - ast_ok} output={output_path}"
    )


if __name__ == "__main__":
    main()
