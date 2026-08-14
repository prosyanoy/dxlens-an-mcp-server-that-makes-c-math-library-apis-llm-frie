"""Transport-independent core: everything the MCP tools do, as plain methods returning dicts."""
from __future__ import annotations

import difflib
from pathlib import Path

from . import snippets
from .compiler import compile_and_run, summarize
from .config import Config
from .docs import count_tokens, render_llms_txt
from .index import ApiIndex
from .parser import parse_header
from .validate import validate_config


class DxLens:
    def __init__(self, cfg: Config | None = None, spec: dict | None = None):
        self.cfg = cfg or Config.from_env()
        self.spec = spec or parse_header(self.cfg.header)
        self.llms_txt = render_llms_txt(self.spec)
        self.index = ApiIndex(self.spec)

    # -- retrieval ---------------------------------------------------------
    def search_api(self, query: str, top_k: int = 5) -> dict:
        hits = self.index.search(query, max(1, min(top_k, 20)))
        return {"query": query, "results": hits,
                "tip": "Call describe_operator(name) for the full record of a hit." if hits else
                       "No matches; try other words (operator, member or constraint names)."}

    def describe_operator(self, name: str) -> dict:
        key = name.strip().split("<")[0].split("::")[-1]
        spec = self.spec
        for op in spec["operators"]:
            if op["name"].lower() == key.lower():
                return {"found": True, "kind": "operator", **op}
        for kind in ("entries", "traits", "functions", "combinators", "enums"):
            for item in spec[kind]:
                if item["name"].lower() == key.lower():
                    return {"found": True, "kind": kind.rstrip("s") if kind != "entries" else "entry", **item}
        entry = spec["entries"][0] if spec["entries"] else None
        if entry:
            members = [m for m in entry["members"] if m["name"].lower() == key.lower()]
            if members:
                return {"found": True, "kind": "member", "entry": entry["name"], "members": members}
        known = [c["name"].split("::")[-1] for c in self.index.chunks if c["kind"] != "constraint"]
        suggestions = difflib.get_close_matches(key, known, n=3, cutoff=0.6)
        suggestions += [h["name"] for h in self.index.search(name, 5) if h["name"] not in suggestions]
        return {"found": False, "error": f"no operator or symbol named '{name}'", "suggestions": suggestions}

    # -- validation / generation ------------------------------------------
    def validate_config(self, config: dict) -> dict:
        return validate_config(self.spec, self.cfg, config)

    def list_templates(self) -> dict:
        return {"templates": snippets.list_templates()}

    def generate_snippet(self, template: str, config: dict, params: dict | None = None, run: bool = True) -> dict:
        rendered = snippets.render(self.spec, template, config, params)
        if "errors" in rendered:
            return {"ok": False, "stage": "generate", "errors": rendered["errors"],
                    "next_step": "Fix the request (see list_templates / describe_operator) and call again."}
        check = summarize(compile_and_run(rendered["code"], self.cfg, run=run), self.spec)
        out = {"ok": check["ok"], "template": template, "code": rendered["code"],
               "description": rendered["description"], "warnings": rendered["warnings"], "compile_check": check}
        if not check["ok"]:
            out["next_step"] = ("The generated code did not compile/run. Use compile_check.errors hints; "
                                "validate_config on the configuration usually pinpoints the cause.")
        return out

    def check_code(self, code: str, run: bool = False) -> dict:
        """Compile-check agent-written code (the retry loop). Running arbitrary code needs DXLENS_ALLOW_RUN=1."""
        do_run = run and self.cfg.allow_run
        out = summarize(compile_and_run(code, self.cfg, run=do_run), self.spec)
        if run and not self.cfg.allow_run:
            out["note"] = "run=true ignored: set DXLENS_ALLOW_RUN=1 to execute agent-written programs."
        return out

    def info(self) -> dict:
        lib = self.spec["library"]
        return {"library": lib["name"], "header": lib["header"], "compiler": self.cfg.cxx,
                "spec_tokens": count_tokens(self.llms_txt),
                "header_tokens": count_tokens(Path(self.cfg.header).read_text())}
