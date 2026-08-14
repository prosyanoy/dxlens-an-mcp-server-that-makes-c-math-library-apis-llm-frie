"""Local BM25 retrieval over spec chunks (one chunk per operator / member / constraint ...)."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .docs import param_form

# Tiny query-side synonym table for common numeric vocabulary.
SYNONYMS = {
    "fp32": ["float", "single"], "fp64": ["double"], "f32": ["float"], "f64": ["double"],
    "points": ["size"], "length": ["size"], "point": ["size"], "len": ["size"],
    "forward": ["direction"], "inverse": ["direction"], "ifft": ["inverse"],
    "batch": ["batched"], "batched": ["batch", "ffts_per_block"],
    "real": ["r2c", "c2r"], "complex": ["c2c"], "gpu": ["sm"], "arch": ["sm"],
    "valid": ["supported", "constraint"], "check": ["supported"],
}


def tokenize(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # split camelCase
    words = re.findall(r"[A-Za-z][A-Za-z0-9]*|\d+", text.replace("_", " "))
    return [w.lower() for w in words]


def build_chunks(spec: dict) -> list[dict]:
    chunks: list[dict] = []

    def add(kind: str, name: str, text: str, ref: dict) -> None:
        chunks.append({"kind": kind, "name": name, "text": text, "ref": ref})

    for e in spec["enums"]:
        add("enum", e["name"], f"enum {e['name']} values {' '.join(e['values'])} {e['doc']}", {"enum": e["name"]})
    for op in spec["operators"]:
        extra = " ".join(c["message"] for c in op["constraints"])
        add("operator", op["name"],
            f"{op['name']} operator {param_form(op)} {op['category']} {op['doc']} {' '.join(op['examples'])} "
            f"{'required' if op['required'] else 'optional'} default {op['default'] or ''} {extra}",
            {"operator": op["name"]})
    for entry in spec["entries"]:
        add("entry", entry["name"], f"{entry['name']} {entry['doc']} {' '.join(entry['examples'])}", {"entry": entry["name"]})
        for m in entry["members"]:
            add("member", f"{entry['name']}::{m['name']}",
                f"{entry['name']} member {m['name']} {m.get('signature', m.get('type', ''))} {m['doc']}",
                {"entry": entry["name"], "member": m["name"]})
        for c in entry["constraints"]:
            add("constraint", c["id"], f"constraint {c['id']} {c['message']} {' '.join(c['operators'])}",
                {"entry": entry["name"], "constraint": c["id"]})
    for t in spec["traits"]:
        add("trait", t["name"], f"trait {t['name']} {t['doc']} {' '.join(t['examples'])}", {"trait": t["name"]})
    for f in spec["functions"] + spec["combinators"]:
        add("function", f["name"], f"function {f['name']} {f['signature']} {f['doc']} {' '.join(f['examples'])}", {"function": f["name"]})
    return chunks


class ApiIndex:
    def __init__(self, spec: dict):
        self.chunks = build_chunks(spec)
        self._docs = [tokenize(c["name"] + " " + c["text"]) for c in self.chunks]
        self._bm25 = BM25Okapi(self._docs)

    def expand(self, query: str) -> list[str]:
        toks = tokenize(query)
        extra: list[str] = []
        for t in toks:
            extra += SYNONYMS.get(t, [])
        return toks + [x for e in extra for x in tokenize(e)]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.expand(query)
        if not q:
            return []
        scores = self._bm25.get_scores(q)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        hits = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            hits.append({"kind": c["kind"], "name": c["name"], "score": round(float(scores[i]), 3),
                         "summary": c["text"][:240], "ref": c["ref"]})
        return hits
