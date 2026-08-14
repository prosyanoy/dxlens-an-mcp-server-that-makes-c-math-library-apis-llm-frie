"""Offline evaluation: raw header vs. DxLens spec as context, with and without the compile-feedback loop.

A task passes only if the program compiles, runs (exit 0) and prints EXACTLY the expected stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..compiler import compile_and_run, summarize
from ..config import Config
from ..docs import count_tokens, render_llms_txt
from ..parser import parse_header
from .knowledge import truncate
from .models import LlamaCppModel, OllamaModel, ScriptedModel
from .tasks import Task, all_tasks

DEFAULT_BUDGETS: list[int | None] = [None, 3000, 1500, 750]


class Runner:
    def __init__(self, cfg: Config, spec: dict, workers: int = 8):
        self.cfg, self.spec, self.workers = cfg, spec, workers
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()

    def execute(self, code: str) -> dict:
        key = hashlib.sha256(code.encode()).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
        if hit is None:
            hit = compile_and_run(code, self.cfg, run=True)
            with self._lock:
                self._cache[key] = hit
        return hit

    def attempt_chain(self, task: Task, model, context: str, feedback: bool, max_retries: int) -> dict:
        history: list[tuple[str, dict]] = []
        attempts, first_ok = 0, None
        while True:
            code = model.generate(task.prompt, context, history)
            res = self.execute(code)
            attempts += 1
            compiled = res["stage"] == "run"
            run_ok = compiled and res["ok"]
            passed = bool(run_ok and res["run"]["stdout"] == task.expected)
            if first_ok is None:
                first_ok = {"compiled": compiled, "passed": passed}
            if passed or not feedback or compiled or attempts > max_retries:
                stage = "pass" if passed else "compile" if not compiled else "runtime" if not run_ok else "wrong_output"
                return {"task": task.id, "kind": task.kind, "passed": passed, "attempts": attempts,
                        "first_try": first_ok, "failure": None if passed else stage}
            history.append((code, summarize(res, self.spec)))

    def run(self, task_list: list[Task], model_factory, contexts: dict[str, str], budgets, max_retries: int = 2) -> list[dict]:
        jobs = []
        for cname, ctext in contexts.items():
            for budget in budgets:
                ctx = truncate(ctext, budget)
                for fb in (False, True):
                    for t in task_list:
                        jobs.append((cname, budget, fb, t, ctx))
        def work(j):
            cname, budget, fb, t, ctx = j
            r = self.attempt_chain(t, model_factory(cname), ctx, fb, max_retries)
            return {"context": cname, "budget": budget, "feedback": fb, "context_tokens": count_tokens(ctx), **r}
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(work, jobs))


def summarize_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["context"], r["budget"]), []).append(r)
    out = []
    for (cname, budget), g in groups.items():
        no_fb = [r for r in g if not r["feedback"]]
        fb = [r for r in g if r["feedback"]]
        n = len(no_fb)
        out.append({"context": cname, "budget": budget, "tokens": no_fb[0]["context_tokens"], "tasks": n,
                    "compile_pass": sum(r["first_try"]["compiled"] for r in no_fb),
                    "pass_at_1": sum(r["passed"] for r in no_fb),
                    "pass_with_loop": sum(r["passed"] for r in fb)})
    return out


def render_markdown(summary: list[dict], model_name: str) -> str:
    def pct(a, n): return f"{100 * a / n:.0f}% ({a}/{n})"
    lines = [f"Model: `{model_name}`; 30 tasks; pass = compiles + runs + exact stdout.", "",
             "| Context | Budget (tokens) | Context size | Compiles | Pass@1 | Pass with DxLens compile loop (<=2 retries) |",
             "|---|---|---|---|---|---|"]
    names = {"raw": "raw header", "spec": "DxLens spec (llms.txt)"}
    for s in summary:
        b = "unlimited" if s["budget"] is None else str(s["budget"])
        lines.append(f"| {names.get(s['context'], s['context'])} | {b} | {s['tokens']} | {pct(s['compile_pass'], s['tasks'])} "
                     f"| {pct(s['pass_at_1'], s['tasks'])} | {pct(s['pass_with_loop'], s['tasks'])} |")
    return "\n".join(lines) + "\n"


def oracle_facts_model():
    """Model that knows the true API (full spec facts): proves every task is solvable and deterministic."""
    from .knowledge import facts_from_context
    from .models import write_program

    class Oracle:
        name = "oracle"
        def __init__(self, ctx): self.ctx = ctx
        def generate(self, prompt, context, history):
            return write_program(prompt, facts_from_context(self.ctx, "spec"))
    return Oracle


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dxlens eval")
    ap.add_argument("--model", choices=["scripted", "ollama", "llamacpp"], default="scripted")
    ap.add_argument("--budgets", default="unlimited,3000,1500,750", help="comma list of token budgets")
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks (smoke test)")
    ap.add_argument("--out", default="docs/eval")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    cfg = Config.from_env()
    spec = parse_header(cfg.header)
    spec_text = render_llms_txt(spec)
    raw = f"// {cfg.header.name}\n" + cfg.header.read_text()
    contexts = {"raw": raw, "spec": spec_text}
    budgets = [None if b == "unlimited" else int(b) for b in args.budgets.split(",")]
    tasks = all_tasks()[: args.limit or None]
    runner = Runner(cfg, spec, args.workers)

    if args.model == "scripted":
        factory = lambda cname: ScriptedModel(cname)
        mname = "scripted (deterministic, context-grounded)"
    elif args.model == "ollama":
        m = OllamaModel(); factory = lambda cname: m; mname = m.name
    else:
        m = LlamaCppModel(); factory = lambda cname: m; mname = m.name

    # sanity: the oracle must solve every task, otherwise the benchmark itself is broken
    oracle = oracle_facts_model()
    orows = [runner.attempt_chain(t, oracle(spec_text), "", False, 0) for t in tasks]
    bad = [r["task"] for r in orows if not r["passed"]]
    if bad:
        print("benchmark broken, oracle fails:", bad)
        return 2
    print(f"oracle solves {len(orows)}/{len(orows)} tasks")

    rows = runner.run(tasks, factory, contexts, budgets)
    summary = summarize_rows(rows)
    md = render_markdown(summary, mname)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"results-{args.model}.json").write_text(json.dumps({"model": mname, "summary": summary, "rows": rows}, indent=1))
    (out / f"results-{args.model}.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
