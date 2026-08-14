"""Compile-check loop: compile (and optionally run) code, return agent-usable diagnostics."""
from __future__ import annotations

import difflib
import os
import re
import subprocess
import tempfile
from pathlib import Path

from .config import Config

DIAG_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): (?P<sev>fatal error|error|warning|note): (?P<msg>.*)$")
TRACE_RE = re.compile(r"^(?P<file>[^\s:][^:]*):(?P<line>\d+):(?:\d+:)?\s+(?:required from|in instantiation of)", re.I)
MAX_OUT = 4000


def parse_diagnostics(stderr: str, user_file: str = "main.cpp") -> list[dict]:
    """Parse gcc/clang text diagnostics into records; notes attach to the preceding error."""
    lines = stderr.replace("\u2018", "'").replace("\u2019", "'").splitlines()
    diags: list[dict] = []
    pending_trace: list[dict] = []
    cur: dict | None = None
    for i, ln in enumerate(lines):
        m = DIAG_RE.match(ln)
        if m:
            cur = {"severity": m["sev"].replace("fatal error", "error"), "file": Path(m["file"]).name,
                   "line": int(m["line"]), "column": int(m["col"]), "message": m["msg"].strip(),
                   "code": "", "trace": pending_trace, "notes": []}
            pending_trace = []
            # source excerpt: gcc prints " 12 | code", clang prints the raw line
            for nxt in lines[i + 1:i + 3]:
                g = re.match(r"^\s*\d+ \|\s?(.*)$", nxt)
                if g:
                    cur["code"] = g.group(1).strip()
                    break
                if nxt and not DIAG_RE.match(nxt) and "^" not in nxt and not nxt.lstrip().startswith("|"):
                    cur["code"] = nxt.strip()
                    break
            if cur["severity"] == "note" and diags:
                diags[-1]["notes"].append(f"{cur['file']}:{cur['line']}: {cur['message']}")
            else:
                diags.append(cur)
            continue
        t = TRACE_RE.match(ln)
        if t:  # gcc prints the instantiation trace BEFORE the error it explains
            pending_trace.append({"file": Path(t["file"]).name, "line": int(t["line"])})
    for d in diags:
        user = [t for t in d["trace"] if t["file"] == user_file]
        d["user_line"] = d["line"] if d["file"] == user_file else (user[-1]["line"] if user else None)
        d["where"] = "user" if d["file"] == user_file else "library"
        d["trace"] = d["trace"][:4]
        d["notes"] = [n for n in d["notes"] if "evaluates to false" not in n][:6]
    return diags


def compile_and_run(code: str, cfg: Config, run: bool = False, stdin: str = "") -> dict:
    """Compile `code` as main.cpp. Returns a plain dict (JSON-serialisable)."""
    if not cfg.cxx:
        return {"ok": False, "stage": "setup", "errors": [], "warnings": [],
                "raw": "no C++ compiler found (install g++ or clang++, or set DXLENS_CXX)",
                "run": None, "compiler": None}
    with tempfile.TemporaryDirectory(prefix="dxlens-") as tmp:
        src, exe = Path(tmp) / "main.cpp", Path(tmp) / "main.out"
        src.write_text(code)
        cmd = [cfg.cxx, *cfg.cxxflags, "-fno-diagnostics-color", "-fmessage-length=0",
               *[f"-I{p}" for p in cfg.include_dirs]]
        cmd += ["-o", str(exe), str(src)] if run else ["-fsyntax-only", str(src)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.compile_timeout, cwd=tmp,
                                  env={**os.environ, "LC_ALL": "C"})
        except subprocess.TimeoutExpired:
            return {"ok": False, "stage": "compile", "errors": [], "warnings": [],
                    "raw": f"compile timed out after {cfg.compile_timeout}s", "run": None, "compiler": cfg.cxx}
        diags = parse_diagnostics(proc.stderr)
        result = {"ok": proc.returncode == 0, "stage": "compile",
                  "errors": [d for d in diags if d["severity"] == "error"],
                  "warnings": [d for d in diags if d["severity"] == "warning"],
                  "raw": proc.stderr[:MAX_OUT], "run": None, "compiler": cfg.cxx}
        if proc.returncode != 0 or not run:
            return result
        result["stage"] = "run"
        try:
            r = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True,
                               timeout=cfg.run_timeout, cwd=tmp)
            result["run"] = {"returncode": r.returncode, "stdout": r.stdout[:MAX_OUT],
                             "stderr": r.stderr[:MAX_OUT], "timed_out": False}
            result["ok"] = r.returncode == 0
        except subprocess.TimeoutExpired as e:
            result["run"] = {"returncode": None, "stdout": (e.stdout or b"").decode(errors="replace")[:MAX_OUT]
                             if isinstance(e.stdout, bytes) else (e.stdout or "")[:MAX_OUT],
                             "stderr": "", "timed_out": True}
            result["ok"] = False
        return result


# ------------------------------------------------------------------- hints
def _all_names(spec: dict) -> dict[str, str]:
    names = {}
    for key in ("operators", "traits", "functions", "enums", "entries"):
        for item in spec.get(key, []):
            names[item["name"]] = key
    return names


def _static_assert_message(msg: str) -> str | None:
    m = re.search(r"static assertion failed(?: due to requirement '.*?')?:\s*(.*)$", msg)
    return m.group(1).strip() if m else None


def add_hints(diags: list[dict], spec: dict) -> list[dict]:
    """Attach `hint` / `spec_ref` to each diagnostic using the parsed API spec."""
    ns = spec["library"]["namespace"]
    names = _all_names(spec)
    entry = spec["entries"][0] if spec["entries"] else None
    for d in diags:
        msg, hint, ref = d["message"], None, None
        sa = _static_assert_message(msg)
        if sa and entry:
            con = next((c for c in entry["constraints"] if c["message"] == sa or sa.startswith(c["message"])), None)
            if con:
                ref = {"constraint": con["id"], "operators": con["operators"]}
                hint = f"Violated constraint {con['id']}: {con['message']}."
                for opn in con["operators"]:
                    op = next(o for o in spec["operators"] if o["name"] == opn)
                    dom = f" valid range {op['range']}" if op["range"] else ""
                    hint += f" See describe_operator('{opn}'){dom}."
                if "only available for" in sa:
                    hint += " Use the execute overload that matches Type<>: " + "; ".join(
                        m["signature"] for m in entry["members"] if m["name"].startswith("execute")) + "."
        else:
            undecl = (re.search(r"'(\w+)' was not declared in this scope", msg)
                      or re.search(r"use of undeclared identifier '(\w+)'", msg)
                      or re.search(r"'(\w+)' (?:has not been declared|is not a member of '[\w:]+')", msg)
                      or re.search(r"no member named '(\w+)' in namespace", msg)
                      or re.search(r"unknown type name '(\w+)'", msg))
            nomember = re.search(r"(?:has no member named|no member named) '(\w+)'", msg)
            if undecl and "namespace" not in msg.split("'")[0]:
                ident = undecl.group(1)
                if ident in names:
                    hint = (f"'{ident}' is declared in namespace {ns}. Qualify it as {ns}::{ident} "
                            f"or add `using namespace {ns};`.")
                    ref = {names[ident]: ident}
                else:
                    close = difflib.get_close_matches(ident, list(names), n=3, cutoff=0.5)
                    if ident == ns or ident.lower() in ("cufftdx", "cublasdx", "dx"):
                        hint = f"The namespace of this library is `{ns}` (not `{ident}`); use `#include <{spec['library']['header']}>` and `using namespace {ns};`."
                    elif close:
                        hint = f"Unknown name '{ident}'. Did you mean: {', '.join(close)}? (all are in namespace {ns})"
                    else:
                        hint = f"Unknown name '{ident}'. Use search_api to find the right identifier; names live in namespace {ns}."
            elif nomember and entry:
                members = [m for m in entry["members"] if m["kind"] == "static_member"]
                hint = (f"No member '{nomember.group(1)}'. Available static members of {entry['name']}: "
                        + "; ".join(f"{m['name']} - {m['doc']}" for m in members if m["doc"]))
                ref = {"entry": entry["name"]}
            elif "no matching function for call to" in msg or "no matching member function" in msg:
                if entry:
                    hint = ("No overload matches. Available: "
                            + "; ".join(m["signature"] for m in entry["members"] if m["kind"] == "method")
                            + ". Pick the execute overload that matches Type<> (c2c / r2c / c2r).")
                    ref = {"entry": entry["name"]}
            elif re.search(r"file not found|No such file or directory", msg):
                hint = f"Include the library with `{spec['library']['include']}`."
            elif "too few template arguments" in msg or "wrong number of template arguments" in msg \
                    or "template argument" in msg:
                hint = "Check each operator's template parameters with describe_operator; operators are default-constructed objects, e.g. Size<256>()."
        d["hint"] = hint
        d["spec_ref"] = ref
    return diags


def summarize(result: dict, spec: dict, max_errors: int = 5) -> dict:
    """Agent-facing view of a compile_and_run result, with hints and a retry prompt."""
    errors = add_hints(result["errors"], spec)[:max_errors]
    out = {"ok": result["ok"], "stage": result["stage"], "compiler": result["compiler"],
           "errors": [{k: v for k, v in e.items() if k != "severity"} for e in errors],
           "warnings": [f"{w['file']}:{w['line']}: {w['message']}" for w in result["warnings"][:3]],
           "run": result["run"]}
    if not result["ok"] and result["stage"] == "compile" and not errors and result["raw"]:
        out["raw"] = result["raw"][:1500]
    if not result["ok"] and errors:
        lines = ["Compilation failed. Fix these and call again:"]
        for e in errors:
            loc = f"line {e['user_line']}" if e.get("user_line") else f"{e['file']}:{e['line']}"
            lines.append(f"- {loc}: {e['message']}" + (f"  HINT: {e['hint']}" if e.get("hint") else ""))
        out["retry_prompt"] = "\n".join(lines)
    return out
