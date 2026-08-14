"""What a context-grounded model can learn from text: scrape facts from raw header OR llms.txt.

Two independent scrapers (C++ source vs. markdown) feed one `Facts` record. Anything the
context does not reveal keeps a PRIOR: the cuFFTDx-flavoured guess an LLM tends to make
(wrong namespace, 2-argument execute, no non-asserting trait ...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from ..docs import count_tokens


@dataclass(frozen=True)
class Facts:
    ns: str = "cufftdx"
    include: str = "minifftdx.hpp"
    entry: str = "FFT"
    enum_type: str = "fft_type"
    enum_dir: str = "fft_direction"
    ops: tuple = (("size", "Size"), ("precision", "Precision"), ("type", "Type"), ("direction", "Direction"),
                  ("ept", "ElementsPerThread"), ("fpb", "FFTsPerBlock"), ("sm", "SM"))
    exec_c2c: str = "smem"      # 'inplace' | 'smem'   (prior: cuFFTDx execute(data, shared))
    exec_r2c: str = "smem"      # 'two' | 'smem'
    exec_c2r: str = "smem"
    half_spectrum: bool = False
    batch_method: str | None = None
    prop_threads: str = "block_dim.x"
    prop_shared: str = "shared_memory_size"
    trait: str | None = None
    learned: frozenset = field(default_factory=frozenset)

    def op(self, key: str) -> str:
        return dict(self.ops)[key]


OP_KEYWORDS = [("precision", r"floating point type"), ("type", r"kind of transform"),
               ("direction", r"direction of the transform"), ("size", r"transform length"),
               ("ept", r"elements each"), ("fpb", r"independent ffts"), ("sm", r"target gpu architecture")]


def _classify_operator(doc: str) -> str | None:
    d = doc.lower()
    return next((k for k, rx in OP_KEYWORDS if re.search(rx, d)), None)


def _classify_exec(name: str, params: str) -> str | None:
    p = re.sub(r"\s+", " ", params).replace(" *", "*").strip()
    if name == "execute_batch":
        return "batch"
    if name != "execute":
        return None
    if "," not in p:
        return "c2c"
    return "r2c" if p.startswith("const real_type") else "c2r" if p.startswith("const value_type") else None


def _items_raw(text: str) -> dict:
    doc = r"((?:[ \t]*///[^\n]*\n)+)"
    clean = lambda d: " ".join(re.sub(r"^\s*///\s?", "", ln) for ln in d.strip().splitlines())
    out: dict = {"ops": [], "methods": [], "members": [], "structs": []}
    if m := re.search(r"^namespace (?!detail)(\w+)\s*\{", text, re.M):
        out["ns"] = m[1]
    if m := re.search(r"template\s*<\s*class Desc\s*>\s*struct (\w+)\s*\{", text):
        out["entry"] = m[1]
    out["enums"] = {m[1]: m[2].replace(" ", "").split(",") for m in re.finditer(r"enum class (\w+)\s*\{([^}]*)\}", text)}
    for m in re.finditer(doc + r"\s*(?:template\s*<[^>]*>\s*)?struct (\w+)\s*:\s*detail::operator_tag", text):
        out["ops"].append((m[2], clean(m[1])))
    for m in re.finditer(doc + r"\s*void (execute\w*)\(([^)]*)\) const", text):
        out["methods"].append((m[2], m[3], clean(m[1])))
    for m in re.finditer(doc + r"\s*static constexpr [\w ]+? (\w+)\s*=", text):
        out["members"].append((m[2], clean(m[1])))
    for m in re.finditer(doc + r"\s*template\s*<[^>]*>\s*struct (\w+)", text):
        out["structs"].append((m[2], clean(m[1])))
    out["half"] = bool(re.search(r"size\s*/\s*2\s*\+\s*1", text))
    return out


def _items_spec(text: str) -> dict:
    out: dict = {"ops": [], "methods": [], "members": [], "structs": []}
    if m := re.search(r"Namespace: `(\w+)`", text):
        out["ns"] = m[1]
    if m := re.search(r"Include: `#include <([^>]+)>`", text):
        out["include"] = m[1]
    if m := re.search(r"^## (\w+)<Desc>", text, re.M):
        out["entry"] = m[1]
    out["enums"] = {m[1]: [v.strip() for v in m[2].split("|")]
                    for m in re.finditer(r"^- `(\w+)`: ([\w ]+(?: \| \w+)+)\.", text, re.M)}
    for m in re.finditer(r"^### `(\w+)(?:<[^`]*>)?\(\)`[^\n]*\n([^\n]*)", text, re.M):
        out["ops"].append((m[1], m[2]))
    for m in re.finditer(r"^- `void (execute\w*)\(([^)]*)\) const`: ([^\n]*)", text, re.M):
        out["methods"].append((m[1], m[2], m[3]))
    for m in re.finditer(r"^- `(\w+)` \([\w ]+\): ([^\n]*)", text, re.M):
        out["members"].append((m[1], m[2]))
    for m in re.finditer(r"^- `(\w+)<Desc>::value`: ([^\n]*)", text, re.M):
        out["structs"].append((m[1], m[2]))
    out["half"] = bool(re.search(r"size\s*/\s*2\s*\+\s*1", text))
    return out


def facts_from_items(items: dict, base: Facts | None = None, header_name: str | None = None) -> Facts:
    f = base or Facts()
    learned = set(f.learned)
    kw: dict = {}
    for k in ("ns", "entry", "include"):
        if k in items:
            kw[k] = items[k]; learned.add(k)
    if header_name and "include" not in items:
        kw["include"] = header_name
    ops = dict(f.ops)
    for name, doc in items["ops"]:
        if key := _classify_operator(doc):
            ops[key] = name; learned.add(f"op:{key}")
    kw["ops"] = tuple(ops.items())
    for typ, key in (("fft_type", "enum_type"), ("fft_direction", "enum_dir")):
        if typ in items["enums"]:
            kw[key] = typ; learned.add(key)
    for name, params, _doc in items["methods"]:
        c = _classify_exec(name, params)
        if c == "c2c": kw["exec_c2c"] = "inplace"; learned.add("exec_c2c")
        elif c == "r2c": kw["exec_r2c"] = "two"; learned.add("exec_r2c")
        elif c == "c2r": kw["exec_c2r"] = "two"; learned.add("exec_c2r")
        elif c == "batch": kw["batch_method"] = name; learned.add("batch")
    for name, doc in items["members"]:
        d = doc.lower()
        if "threads needed" in d: kw["prop_threads"] = name; learned.add("prop_threads")
        if "shared memory bytes" in d: kw["prop_shared"] = name; learned.add("prop_shared")
    for name, doc in items["structs"]:
        if "non-asserting" in doc.lower(): kw["trait"] = name; learned.add("trait")
    if items.get("half"):
        kw["half_spectrum"] = True; learned.add("half_spectrum")
    return replace(f, learned=frozenset(learned), **kw)


def facts_from_context(text: str, mode: str) -> Facts:
    """mode: 'raw' (C++ header text) or 'spec' (llms.txt)."""
    if mode == "raw":
        return facts_from_items(_items_raw(text), header_name="minifftdx.hpp")
    return facts_from_items(_items_spec(text))


def revise_facts(f: Facts, feedback: dict) -> Facts:
    """Learn from DxLens compile feedback (hints only; plain compiler text teaches nothing here)."""
    hints = " ".join(e.get("hint") or "" for e in feedback.get("errors", []))
    kw = {}
    if m := re.search(r"namespace(?: of this library is)? `?(\w+)`?", hints):
        if m[1] not in ("of", "this"):
            kw["ns"] = m[1]
    methods = [(m[1], m[2], "") for m in re.finditer(r"void (execute\w*)\(([^)]*)\) const", hints)]
    if methods:
        f2 = facts_from_items({"ops": [], "methods": methods, "members": [], "structs": [], "enums": {}}, base=f)
        kw.update(exec_c2c=f2.exec_c2c, exec_r2c=f2.exec_r2c, exec_c2r=f2.exec_c2r, batch_method=f2.batch_method)
    if "Available static members" in hints:
        mem = [(m[1].strip(), m[2]) for m in re.finditer(r"(?:^|; |: )(\w+) - ([^;]*)", hints.split("Available static members")[1])]
        f2 = facts_from_items({"ops": [], "methods": [], "members": mem, "structs": [], "enums": {}}, base=f)
        kw.update(prop_threads=f2.prop_threads, prop_shared=f2.prop_shared)
    return replace(f, **kw)


def truncate(text: str, budget: int | None) -> str:
    """Keep whole lines until the token budget is spent (what a small context window would see)."""
    if budget is None or count_tokens(text) <= budget:
        return text
    out, used = [], 0
    for ln in text.splitlines():
        n = count_tokens(ln)
        if used + n > budget:
            break
        out.append(ln); used += n
    return "\n".join(out) + "\n"
