"""Models for the harness: a deterministic scripted one plus optional local LLM backends."""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .knowledge import Facts, facts_from_context, revise_facts


# ---------------------------------------------------------------- prompt parsing
def parse_prompt(prompt: str) -> dict:
    """Extract the task slots a human (or model) would read from the natural-language prompt."""
    def num(rx, default=None):
        m = re.search(rx, prompt)
        return int(m[1]) if m else default
    kind = ("supported" if "non-asserting" in prompt else
            "info" if "compile-time properties" in prompt else
            "batch" if "in one batch" in prompt else
            "c2r" if "complex-to-real" in prompt else
            "r2c" if "real-to-complex FFT of x[n] = sin" in prompt else
            "impulse" if "unit impulse" in prompt else
            "roundtrip" if "then the inverse FFT" in prompt else "peak")
    return {"kind": kind, "N": num(r"(\d+)-point"), "T": "double" if "fp64" in prompt else "float",
            "f": num(r"2\*pi\*\(?(\d+)"), "E": num(r"(\d+) elements per thread"),
            "S": num(r"SM (\d+)"), "F": num(r"(\d+) FFTs per block"), "B": num(r"runs (\d+) independent")}


# ----------------------------------------------------------------- code writing
PEAK_LOOP = """  unsigned peak = 0; double best = std::abs({a}[0]);
  for (unsigned k = 1; k < {n}; ++k) {{ double m = std::abs({a}[k]); if (m > best + 1e-3) {{ best = m; peak = k; }} }}
"""


def _desc(s: dict, f: Facts, type_: str, direction: str | None) -> str:
    parts = [f"{f.op('size')}<{s['N']}>()", f"{f.op('precision')}<{s['T']}>()", f"{f.op('type')}<{f.enum_type}::{type_}>()"]
    if direction:
        parts.append(f"{f.op('direction')}<{f.enum_dir}::{direction}>()")
    if s.get("E"): parts.append(f"{f.op('ept')}<{s['E']}>()")
    if s.get("F") and s["kind"] != "supported": parts.append(f"{f.op('fpb')}<{s['F']}>()")
    if s.get("S"): parts.append(f"{f.op('sm')}<{s['S']}>()")
    if s["kind"] == "supported" and s.get("F"): parts.append(f"{f.op('fpb')}<{s['F']}>()")
    parts.append("Block()")
    return " + ".join(parts)


def _call(f: Facts, kind: str, args: list[str]) -> str:
    style = {"c2c": f.exec_c2c, "r2c": f.exec_r2c, "c2r": f.exec_c2r}[kind]
    if style == "smem":  # cuFFTDx habit: trailing shared-memory pointer
        return f"Fft().execute({args[0]}, nullptr);" if kind == "c2c" else f"Fft().execute({args[0]}, {args[1]}, nullptr);"
    return f"Fft().execute({', '.join(args)});"


def write_program(prompt: str, f: Facts) -> str:
    s = parse_prompt(prompt)
    k, n, t = s["kind"], s["N"], s["T"]
    head = f"#include <cstdio>\n#include <cmath>\n#include <complex>\n#include <vector>\n#include <{f.include}>\nusing namespace {f.ns};\n"
    pi = "3.14159265358979323846"
    body = ""
    if k in ("peak", "impulse", "roundtrip", "batch"):
        direction = "forward"
        if k == "roundtrip":
            body = (f"  using Fwd = {f.entry}<decltype({_desc(s, f, 'c2c', 'forward')})>;\n"
                    f"  using Inv = {f.entry}<decltype({_desc(s, f, 'c2c', 'inverse')})>;\n"
                    f"  std::complex<{t}> data[{n}];\n"
                    f"  for (int i = 0; i < {n}; ++i) data[i] = std::complex<{t}>(({'i % 7'}) - 3, 0);\n"
                    f"  {_call(f, 'c2c', ['data']).replace('Fft()', 'Fwd()')}\n  {_call(f, 'c2c', ['data']).replace('Fft()', 'Inv()')}\n"
                    f"  double sum = 0; for (int i = 0; i < {n}; ++i) sum += data[i].real() / {n};\n"
                    f'  std::printf("sum=%.3f\\n", sum);\n')
        else:
            body = f"  using Fft = {f.entry}<decltype({_desc(s, f, 'c2c', direction)})>;\n"
            if k == "batch":
                b = s["B"]
                call = (f"Fft().{f.batch_method}(data.data(), {b});" if f.batch_method else
                        f"for (int b = 0; b < {b}; ++b) {{ {_call(f, 'c2c', [f'data.data() + b * {n}'])} }}")
                body += (f"  std::vector<std::complex<{t}>> data({n} * {b});\n"
                         f"  for (int b = 0; b < {b}; ++b) for (int i = 0; i < {n}; ++i)\n"
                         f"    data[b * {n} + i] = std::complex<{t}>(std::cos(2.0 * {pi} * (b + 1) * i / {n}), 0);\n  {call}\n"
                         f'  std::printf("peaks=");\n  for (int b = 0; b < {b}; ++b) {{\n'
                         + PEAK_LOOP.format(a=f"(data.data() + b * {n})", n=n).replace("\n", "\n  ")
                         + '  std::printf("%s%u", b ? "," : "", peak); }\n  std::printf("\\n");\n')
            elif k == "impulse":
                body += (f"  std::complex<{t}> data[{n}] = {{}};\n  data[0] = 1;\n  {_call(f, 'c2c', ['data'])}\n"
                         f"  double s = 0; for (int i = 0; i < {n}; ++i) s += data[i].real();\n"
                         '  std::printf("sum_re=%.3f\\n", s);\n')
            else:
                body += (f"  std::complex<{t}> data[{n}];\n"
                         f"  for (int i = 0; i < {n}; ++i) data[i] = std::complex<{t}>(std::cos(2.0 * {pi} * {s['f']} * i / {n}), 0);\n"
                         f"  {_call(f, 'c2c', ['data'])}\n" + PEAK_LOOP.format(a="data", n=n)
                         + '  std::printf("peak_bin=%u peak_mag=%.3f\\n", peak, best);\n')
    elif k == "r2c":
        out = n // 2 + 1 if f.half_spectrum else n
        body = (f"  using Fft = {f.entry}<decltype({_desc(s, f, 'r2c', None)})>;\n"
                f"  {t} in[{n}]; std::complex<{t}> out[{out}];\n"
                f"  for (int i = 0; i < {n}; ++i) in[i] = std::sin(2.0 * {pi} * {s['f']} * i / {n});\n"
                f"  {_call(f, 'r2c', ['in', 'out'])}\n" + PEAK_LOOP.format(a="out", n=out)
                + f'  std::printf("bins=%d peak_bin=%u peak_mag=%.3f\\n", {out}, peak, best);\n')
    elif k == "c2r":
        out = n // 2 + 1 if f.half_spectrum else n
        body = (f"  using Fwd = {f.entry}<decltype({_desc(s, f, 'r2c', None)})>;\n"
                f"  using Inv = {f.entry}<decltype({_desc(s, f, 'c2r', None)})>;\n"
                f"  {t} in[{n}], back[{n}]; std::complex<{t}> spec[{out}];\n"
                f"  for (int i = 0; i < {n}; ++i) in[i] = std::cos(2.0 * {pi} * {s['f']} * i / {n});\n"
                f"  {_call(f, 'r2c', ['in', 'spec']).replace('Fft()', 'Fwd()')}\n"
                f"  {_call(f, 'c2r', ['spec', 'back']).replace('Fft()', 'Inv()')}\n"
                f'  std::printf("x1=%.3f\\n", back[1] / {n});\n')
    elif k == "info":
        body = (f"  using Fft = {f.entry}<decltype({_desc(s, f, 'c2c', 'forward')})>;\n"
                f'  std::printf("threads_per_fft=%u shared_bytes=%u\\n", (unsigned)Fft::{f.prop_threads}, (unsigned)Fft::{f.prop_shared});\n')
    elif k == "supported":
        d = _desc(s, f, "c2c", "forward")
        if f.trait:
            body = f'  constexpr bool ok = {f.trait}<decltype({d})>::value;\n  std::printf("supported=%d\\n", ok ? 1 : 0);\n'
        else:  # no way to ask non-destructively: instantiate and assume it is fine
            body = f'  using Fft = {f.entry}<decltype({d})>;\n  (void)sizeof(Fft);\n  std::printf("supported=1\\n");\n'
    return head + "int main() {\n" + body + "  return 0;\n}\n"


# ------------------------------------------------------------------------ models
class ScriptedModel:
    """Deterministic stand-in for an LLM: writes code from ONLY the facts its context reveals."""

    def __init__(self, context_mode: str):
        self.mode = context_mode
        self.name = f"scripted/{context_mode}"

    def generate(self, prompt: str, context: str, history: list[tuple[str, dict]]) -> str:
        facts = facts_from_context(context, self.mode)
        for _code, feedback in history:
            facts = revise_facts(facts, feedback)
        return write_program(prompt, facts)


def extract_code(text: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+)?\s*\n(.*?)```", text, re.S)
    return (m[1] if m else text).strip() + "\n"


class _HttpModel:
    def build_prompt(self, prompt: str, context: str, history: list[tuple[str, dict]]) -> str:
        p = ("You write C++17 programs. Reply with ONE complete program in a ```cpp block and nothing else.\n\n"
             f"### Library reference\n{context}\n\n### Task\n{prompt}\n")
        if history:
            code, fb = history[-1]
            p += f"\n### Your previous attempt\n```cpp\n{code}```\n### Compiler feedback\n{fb.get('retry_prompt') or json.dumps(fb)[:1500]}\nFix it.\n"
        return p

    def _post(self, url: str, payload: dict) -> dict:
        req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())


class OllamaModel(_HttpModel):
    def __init__(self, url: str | None = None, model: str | None = None):
        self.url = (url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
        self.name = f"ollama/{self.model}"

    def generate(self, prompt, context, history):
        out = self._post(f"{self.url}/api/generate", {"model": self.model, "stream": False,
                         "prompt": self.build_prompt(prompt, context, history), "options": {"temperature": 0, "seed": 0}})
        return extract_code(out["response"])


class LlamaCppModel(_HttpModel):
    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("LLAMACPP_URL", "http://localhost:8080")).rstrip("/")
        self.name = "llama.cpp"

    def generate(self, prompt, context, history):
        out = self._post(f"{self.url}/completion", {"prompt": self.build_prompt(prompt, context, history),
                         "n_predict": 1500, "temperature": 0, "seed": 0})
        return extract_code(out["content"])
