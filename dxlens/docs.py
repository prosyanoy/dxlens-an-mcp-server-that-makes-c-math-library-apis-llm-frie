"""Render a spec as compact, llms.txt-style markdown."""
from __future__ import annotations

import re


def count_tokens(text: str) -> int:
    """Cheap, tokenizer-free estimate (words + punctuation marks); stable and offline."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def param_form(op: dict) -> str:
    """Usage form with placeholders, e.g. Size<N>() or Type<fft_type::c2c|r2c|c2r>()."""
    if not op["template_params"]:
        return f"{op['name']}()"
    args = []
    for p in op["template_params"]:
        if p.get("values"):
            args.append(f"{p['enum']}::" + "|".join(p["values"]))
        elif p["kind"] == "type":
            args.append("T" if p["name"] == "" else p["name"])
        else:
            args.append(p["name"])
    return f"{op['name']}<{', '.join(args)}>()"


def _domain(op: dict) -> str:
    bits = []
    for p in op["template_params"]:
        if p.get("values"):
            bits.append(f"{p['name']}: {p['enum']} in {{{', '.join(p['values'])}}}")
        elif p["kind"] == "non_type":
            d = f"{p['name']}: {p['type']}"
            if op["range"]:
                d += f" in [{op['range'][0]}, {op['range'][1]}]"
            if op["power_of_two"]:
                d += ", power of two"
            bits.append(d)
        elif p["kind"] == "type":
            bits.append(f"{p['name']}: C++ type")
    return "; ".join(bits)


def render_llms_txt(spec: dict) -> str:
    lib = spec["library"]
    ns = lib["namespace"]
    out: list[str] = [f"# {lib['name']}", f"> {lib['summary']}", ""]
    out += [f"Include: `{lib['include']}`", f"Namespace: `{ns}`", ""]

    entry = spec["entries"][0] if spec["entries"] else None
    comb = spec["combinators"][0] if spec["combinators"] else None
    if entry and comb:
        out += [
            "## Workflow",
            f"1. Describe the problem by adding operators with `{comb['name']}`: `A() + B() + ...` (the only way to build a description).",
            f"2. Name its type: `using Desc = decltype(A() + B() + ...);`",
            f"3. Instantiate `{entry['name']}<Desc>` (constraints are checked at compile time) and call its members.",
            "",
        ]

    if spec["enums"]:
        out.append("## Enums")
        for e in spec["enums"]:
            out.append(f"- `{e['name']}`: {' | '.join(e['values'])}. {e['doc']}")
        out.append("")

    out.append("## Operators")
    for op in spec["operators"]:
        flags = [op["category"]]
        if op["required"]:
            flags.append("required")
        if op["default"]:
            flags.append(f"default {op['default']}")
        out.append(f"### `{param_form(op)}` [{', '.join(flags)}]")
        out.append(op["doc"])
        dom = _domain(op)
        if dom:
            out.append(f"- Domain: {dom}")
        for ex in op["examples"]:
            out.append(f"- Example: `{ex}`")
        out.append("")

    if entry:
        out.append(f"## {entry['name']}<Desc>")
        out.append(entry["doc"])
        out.append("")
        out.append("Constraints (compile-time static_assert; message text is what the compiler prints):")
        for c in entry["constraints"]:
            out.append(f"- {c['id']}: {c['message']}")
        out.append("")
        statics = [m for m in entry["members"] if m["kind"] == "static_member"]
        methods = [m for m in entry["members"] if m["kind"] == "method"]
        out.append("Static members:")
        for m in statics:
            out.append(f"- `{m['name']}` ({m['type'].replace('const ', '')}): {m['doc']}")
        out.append("")
        out.append("Methods:")
        for m in methods:
            out.append(f"- `{m['signature']}`: {m['doc']}")
        out.append("")
        out.append("Examples:")
        for ex in entry["examples"]:
            out += ["```cpp", ex, "```"]
        out.append("")

    if spec["traits"] or spec["functions"]:
        out.append("## Helpers")
        for t in spec["traits"]:
            out.append(f"- `{t['name']}<Desc>::value`: {t['doc']}")
            for ex in t["examples"]:
                out.append(f"  - Example: `{ex}`")
        for f in spec["functions"]:
            out.append(f"- `{f['signature']}`: {f['doc']}")
            for ex in f["examples"]:
                out.append(f"  - Example: `{ex}`")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
