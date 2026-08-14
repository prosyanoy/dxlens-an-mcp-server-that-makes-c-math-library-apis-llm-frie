import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dxlens.docs import render_llms_txt
from dxlens.eval.harness import Runner, oracle_facts_model, render_markdown, summarize_rows
from dxlens.eval.knowledge import Facts, facts_from_context, revise_facts, truncate
from dxlens.eval.models import LlamaCppModel, OllamaModel, ScriptedModel, extract_code, parse_prompt, write_program
from dxlens.eval.tasks import all_tasks


@pytest.fixture(scope="module")
def contexts(cfg, spec):
    return {"raw": f"// {cfg.header.name}\n" + cfg.header.read_text(), "spec": render_llms_txt(spec)}


def test_thirty_unique_tasks_with_expected_outputs():
    tasks = all_tasks()
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30
    assert {t.kind for t in tasks} == {"peak", "roundtrip", "impulse", "r2c", "c2r", "batch", "info", "supported"}
    assert all(t.expected.endswith("\n") for t in tasks)
    assert {t.expected for t in tasks if t.kind == "supported"} == {"supported=0\n", "supported=1\n"}


def test_parse_prompt_recovers_slots():
    t = next(x for x in all_tasks() if x.id == "peak-256-3-float-ept4")
    s = parse_prompt(t.prompt)
    assert (s["kind"], s["N"], s["T"], s["f"], s["E"]) == ("peak", 256, "float", 3, 4)
    assert parse_prompt(next(x for x in all_tasks() if x.kind == "supported").prompt)["kind"] == "supported"


def test_both_scrapers_agree_on_full_context(contexts):
    raw, spec = facts_from_context(contexts["raw"], "raw"), facts_from_context(contexts["spec"], "spec")
    assert replace(raw, learned=frozenset()) == replace(spec, learned=frozenset())
    assert spec.ns == "minifftdx" and spec.exec_c2c == "inplace" and spec.trait == "is_supported"
    assert spec.batch_method == "execute_batch" and spec.prop_threads == "threads_per_fft" and spec.half_spectrum


def test_truncation_hides_late_facts(contexts):
    small = facts_from_context(truncate(contexts["raw"], 1500), "raw")
    assert small.trait is None and small.exec_c2c == "smem"  # priors survive when context is cut
    assert facts_from_context(truncate(contexts["spec"], 1500), "spec").exec_c2c == "inplace"


def test_revise_facts_from_dxlens_hints():
    fb = {"errors": [{"hint": "The namespace of this library is `minifftdx` (not `cufftdx`); use ..."},
                     {"hint": "No overload matches. Available: void execute(value_type * data) const; "
                              "void execute(const real_type * in, value_type * out) const; "
                              "void execute_batch(value_type * data, unsigned int batch) const."}]}
    f = revise_facts(Facts(), fb)
    assert f.ns == "minifftdx" and f.exec_c2c == "inplace" and f.exec_r2c == "two" and f.batch_method == "execute_batch"


def test_oracle_solves_every_task(cfg, spec, contexts):
    """Proves each task is solvable and each expected output matches real execution (deterministic)."""
    runner = Runner(cfg, spec, workers=4)
    oracle = oracle_facts_model()(contexts["spec"])
    failures = [t.id for t in all_tasks() if not runner.attempt_chain(t, oracle, "", False, 0)["passed"]]
    assert failures == []


def test_feedback_loop_recovers_raw_failures(cfg, spec, contexts):
    runner = Runner(cfg, spec, workers=4)
    ctx = truncate(contexts["raw"], 1500)
    task = all_tasks()[0]
    no_loop = runner.attempt_chain(task, ScriptedModel("raw"), ctx, False, 2)
    loop = runner.attempt_chain(task, ScriptedModel("raw"), ctx, True, 2)
    assert not no_loop["passed"] and no_loop["failure"] == "compile"
    assert loop["passed"] and loop["attempts"] > 1


def test_wrong_output_is_a_failure(cfg, spec):
    runner = Runner(cfg, spec)

    class Liar:
        def generate(self, *a): return '#include <cstdio>\nint main(){std::puts("peak_bin=1 peak_mag=1.000");}\n'
    r = runner.attempt_chain(all_tasks()[0], Liar(), "", True, 2)
    assert not r["passed"] and r["failure"] == "wrong_output" and r["attempts"] == 1


def test_summary_and_markdown():
    rows = [{"context": "spec", "budget": None, "feedback": fb, "context_tokens": 10, "passed": True,
             "first_try": {"compiled": True, "passed": True}} for fb in (False, True)]
    md = render_markdown(summarize_rows(rows), "m")
    assert "| DxLens spec (llms.txt) | unlimited | 10 | 100% (1/1) | 100% (1/1) | 100% (1/1) |" in md


def test_write_program_is_deterministic():
    t = all_tasks()[3]
    assert write_program(t.prompt, Facts()) == write_program(t.prompt, Facts())


def test_extract_code():
    assert extract_code("x\n```cpp\nint a;\n```\ny") == "int a;\n"
    assert extract_code("int b;") == "int b;\n"


class _Stub(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Stub.seen.append((self.path, body))
        text = "```cpp\nint main(){}\n```"
        out = {"response": text} if self.path == "/api/generate" else {"content": text}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a): pass


def test_local_llm_backends_against_stub_server():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}"
    try:
        hist = [("int x;", {"retry_prompt": "Compilation failed. Fix these"})]
        assert OllamaModel(url, "m").generate("task", "ctx", hist) == "int main(){}\n"
        assert LlamaCppModel(url).generate("task", "ctx", []) == "int main(){}\n"
    finally:
        srv.shutdown(); srv.server_close()
    path, body = _Stub.seen[0]
    assert path == "/api/generate" and "ctx" in body["prompt"] and "Compilation failed" in body["prompt"] and body["options"]["temperature"] == 0
