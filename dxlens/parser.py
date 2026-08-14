"""libclang-based extraction of a Dx-style header into a JSON-serialisable spec.

The parser walks the real C++ AST (so template parameter kinds, signatures and
static_assert constraints are exact) and merges in the author's doc-comment tags
(@dx_operator, @required, @default, @range, @pow2, @example, ...).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import clang.cindex as ci

SPEC_VERSION = 1
K = ci.CursorKind

TEMPLATE_PARAM_KINDS = {
    K.TEMPLATE_TYPE_PARAMETER: "type",
    K.TEMPLATE_NON_TYPE_PARAMETER: "non_type",
    K.TEMPLATE_TEMPLATE_PARAMETER: "template",
}


class ParseError(RuntimeError):
    pass


def _system_include_args() -> list[str]:
    """libclang (pip wheel) ships no compiler headers; borrow the host compiler's."""
    args: list[str] = []
    for cxx in ("g++", "clang++"):
        try:
            out = subprocess.run([cxx, "-print-file-name=include"], capture_output=True,
                                 text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out and Path(out).is_dir():
            args += ["-isystem", out]
            break
    return args


# ---------------------------------------------------------------- doc comments
def parse_doc(raw: str | None) -> dict:
    """Split a doc comment into free text and @tags. Tag values keep line structure."""
    if not raw:
        return {"text": "", "tags": {}, "examples": []}
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip()
        ln = re.sub(r"^(///|//!|/\*\*|\*/|\*)\s?", "", ln) if ln.startswith(("//", "/*", "*")) else ln
        lines.append(ln.rstrip())
    text: list[str] = []
    tags: dict[str, list[str]] = {}
    examples: list[str] = []
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf, current
        if current is None:
            return
        body = "\n".join(buf).strip("\n")
        if current == "example":
            examples.append(body.strip())
        else:
            tags.setdefault(current, []).append(body.strip())
        buf, current = [], None

    for ln in lines:
        m = re.match(r"@(\w+)\s*(.*)$", ln)
        if m:
            flush()
            current, rest = m.group(1), m.group(2)
            buf = [rest] if rest else []
        elif current is not None:
            buf.append(ln)
        else:
            text.append(ln)
    flush()
    return {"text": " ".join(t for t in text if t).strip(), "tags": tags, "examples": examples}


def _tag(doc: dict, name: str) -> str | None:
    vals = doc["tags"].get(name)
    return vals[0] if vals else None


def _has(doc: dict, name: str) -> bool:
    return name in doc["tags"]


# ------------------------------------------------------------- cursor helpers
def _template_params(cursor) -> list[dict]:
    params = []
    for ch in cursor.get_children():
        kind = TEMPLATE_PARAM_KINDS.get(ch.kind)
        if not kind:
            continue
        toks = [t.spelling for t in ch.get_tokens()]
        p = {"name": ch.spelling, "kind": kind, "variadic": "..." in toks}
        if kind == "non_type":
            p["type"] = ch.type.spelling
        elif kind == "template":
            p["type"] = "template"
        else:
            p["type"] = "typename"
        if "=" in toks:
            p["default"] = " ".join(toks[toks.index("=") + 1:])
        params.append(p)
    return params


def _template_header(params: list[dict]) -> str:
    parts = []
    for p in params:
        dots = "..." if p["variadic"] else ""
        if p["kind"] == "non_type":
            parts.append(f"{p['type']} {p['name']}")
        elif p["kind"] == "template":
            parts.append(f"template<...> class{dots} {p['name']}")
        else:
            parts.append(f"class{dots} {p['name']}")
    return "template<" + ", ".join(parts) + ">"


def _static_assert(cursor) -> dict | None:
    toks = list(cursor.get_tokens())
    strs = [i for i, t in enumerate(toks) if t.kind == ci.TokenKind.LITERAL and t.spelling.startswith('"')]
    if not strs:
        return None
    first = strs[0]
    while len(strs) > 1 and strs[1] == strs[0] + 1:  # adjacent literals concatenate
        strs.pop(0)
    # trailing run of string literals before ')'
    end = max(strs)
    start = end
    while start - 1 >= 0 and toks[start - 1].kind == ci.TokenKind.LITERAL and toks[start - 1].spelling.startswith('"'):
        start -= 1
    message = "".join(t.spelling[1:-1] for t in toks[start:end + 1])
    # condition: tokens between the opening '(' and the ',' before the message
    open_idx = next(i for i, t in enumerate(toks) if t.spelling == "(")
    cond = " ".join(t.spelling for t in toks[open_idx + 1:start - 1])
    cond = re.sub(r"\s+(::|,|\(|\)|<|>)\s*", lambda m: m.group(1) if m.group(1) in ("::",) else f" {m.group(1)} ", cond)
    cond = re.sub(r"\s+", " ", cond).strip()
    cond = re.sub(r"\s*(<|>|::)\s*", r"\1", cond)  # tighten template brackets
    cond = re.sub(r"!\s+", "!", cond).replace(">=", " >= ").replace("<=", " <= ")
    return {"condition": cond, "message": message}


def _args(cursor) -> str:
    parms = [c for c in cursor.get_children() if c.kind == K.PARM_DECL]
    return ", ".join(f"{a.type.spelling} {a.spelling}".strip() for a in parms)


def _method(cursor) -> dict:
    args = _args(cursor)
    const = " const" if cursor.is_const_method() else ""
    doc = parse_doc(cursor.raw_comment)
    return {"kind": "method", "name": cursor.spelling,
            "signature": f"{cursor.result_type.spelling} {cursor.spelling}({args}){const}",
            "doc": doc["text"]}


def _member_var(cursor) -> dict:
    doc = parse_doc(cursor.raw_comment)
    return {"kind": "static_member", "name": cursor.spelling, "type": cursor.type.spelling,
            "doc": doc["text"]}


# --------------------------------------------------------------------- driver
def parse_header(path: str | Path, extra_args: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise ParseError(f"header not found: {path}")
    args = ["-x", "c++", "-std=c++17", f"-I{path.parent}"] + _system_include_args() + (extra_args or [])
    index = ci.Index.create()
    tu = index.parse(str(path), args=args)
    fatal = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if fatal:
        raise ParseError("libclang could not parse header: " + "; ".join(d.spelling for d in fatal[:3]))

    text = path.read_text()
    spec: dict = {
        "dxlens_spec_version": SPEC_VERSION,
        "library": {"name": path.stem, "summary": "", "header": path.name, "namespace": "",
                    "include": f"#include <{path.name}>",
                    "sha256": hashlib.sha256(text.encode()).hexdigest()},
        "enums": [], "operators": [], "combinators": [], "entries": [], "traits": [], "functions": [],
    }

    def in_file(c) -> bool:
        return bool(c.location.file) and Path(c.location.file.name).resolve() == path

    def visit_namespace(ns, qual: str) -> None:
        for c in ns.get_children():
            if not in_file(c):
                continue
            if c.kind == K.NAMESPACE:
                if c.spelling == "detail":
                    continue
                doc = parse_doc(c.raw_comment)
                if _has(doc, "dx_library"):
                    head, _, rest = (_tag(doc, "dx_library") or "").partition("\n")
                    spec["library"]["name"] = head.strip() or spec["library"]["name"]
                    spec["library"]["summary"] = " ".join(rest.split())
                    spec["library"]["namespace"] = c.spelling
                visit_namespace(c, c.spelling)
            elif c.kind == K.ENUM_DECL:
                spec["enums"].append(_enum(c))
            elif c.kind in (K.CLASS_TEMPLATE, K.STRUCT_DECL):
                _classify_class(c, spec)
            elif c.kind == K.FUNCTION_TEMPLATE:
                _classify_function(c, spec)

    visit_namespace(tu.cursor, "")
    if not spec["library"]["namespace"] and (spec["operators"] or spec["entries"]):
        raise ParseError("no namespace carrying @dx_library found; annotate the library namespace")
    _link_constraints(spec)
    return spec


def _enum(c) -> dict:
    doc = parse_doc(c.raw_comment)
    values = [e.spelling for e in c.get_children() if e.kind == K.ENUM_CONSTANT_DECL]
    return {"name": c.spelling, "values": values, "doc": doc["text"]}


def _classify_class(c, spec: dict) -> None:
    doc = parse_doc(c.raw_comment)
    params = _template_params(c) if c.kind == K.CLASS_TEMPLATE else []
    tags = doc["tags"]
    if "dx_operator" in tags:
        spec["operators"].append(_operator(c, doc, params, spec))
    elif "dx_entry" in tags:
        spec["entries"].append(_entry(c, doc, params))
    elif "dx_trait" in tags:
        spec["traits"].append({"name": c.spelling, "signature": _template_header(params) + f" struct {c.spelling}",
                               "template_params": params, "doc": doc["text"], "examples": doc["examples"]})


def _operator(c, doc: dict, params: list[dict], spec: dict) -> dict:
    rng = _tag(doc, "range")
    members = [_member_var(m) for m in c.get_children() if m.kind == K.VAR_DECL]
    op = {
        "name": c.spelling,
        "category": _tag(doc, "dx_operator") or "operator",
        "signature": _template_header(params) + f" struct {c.spelling}" if params else f"struct {c.spelling}",
        "template_params": params,
        "doc": doc["text"],
        "required": _has(doc, "required"),
        "default": _tag(doc, "default"),
        "range": [int(x) for x in rng.split()] if rng else None,
        "power_of_two": _has(doc, "pow2"),
        "members": members,
        "examples": doc["examples"],
        "constraints": [],
    }
    for p in params:  # attach enum domain to enum-typed non-type params
        base = re.split(r"::", p.get("type", ""))[-1]
        for e in spec["enums"]:
            if e["name"] == base:
                p["enum"] = e["name"]
                p["values"] = e["values"]
    return op


def _entry(c, doc: dict, params: list[dict]) -> dict:
    members, asserts = [], []
    for ch in c.get_children():
        if ch.kind == K.STATIC_ASSERT:
            sa = _static_assert(ch)
            if sa:
                asserts.append(sa)
        elif ch.kind == K.CXX_METHOD and ch.access_specifier == ci.AccessSpecifier.PUBLIC:
            members.append(_method(ch))
            for inner in ch.walk_preorder():  # usage constraints guarding a specific overload
                if inner.kind == K.STATIC_ASSERT:
                    sa = _static_assert(inner)
                    if sa:
                        sa["method"] = ch.spelling
                        asserts.append(sa)
        elif ch.kind == K.VAR_DECL:
            members.append(_member_var(ch))
    for i, sa in enumerate(asserts, 1):
        sa["id"] = f"C{i}"
    return {"name": c.spelling, "signature": _template_header(params) + f" struct {c.spelling}",
            "template_params": params, "doc": doc["text"], "examples": doc["examples"],
            "members": members, "constraints": asserts}


def _classify_function(c, spec: dict) -> None:
    doc = parse_doc(c.raw_comment)
    params = _template_params(c)
    args = _args(c)
    rec = {"name": c.spelling, "signature": f"{_template_header(params)} {c.result_type.spelling} {c.spelling}({args})",
           "template_params": params, "doc": doc["text"], "examples": doc["examples"]}
    if _has(doc, "dx_combinator"):
        spec["combinators"].append(rec)
    elif _has(doc, "dx_function"):
        spec["functions"].append(rec)


def _link_constraints(spec: dict) -> None:
    """Attach every library constraint to the operators its message names."""
    names = [o["name"] for o in spec["operators"]]
    for entry in spec["entries"]:
        for con in entry["constraints"]:
            con["operators"] = [n for n in names if re.search(rf"\b{re.escape(n)}\b", con["message"])]
            for op in spec["operators"]:
                if op["name"] in con["operators"]:
                    op["constraints"].append({"id": con["id"], "message": con["message"]})
