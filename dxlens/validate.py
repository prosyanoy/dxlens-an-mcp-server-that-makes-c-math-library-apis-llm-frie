"""validate_config: map a JSON-ish configuration to operators, pre-check domains, then ask the compiler."""
from __future__ import annotations

import difflib
import re

from .compiler import compile_and_run, summarize
from .config import Config

TYPE_ALIASES = {"fp32": "float", "f32": "float", "float32": "float", "single": "float",
                "fp64": "double", "f64": "double", "float64": "double"}
KEY_ALIASES = {"elements_per_thread": "ElementsPerThread", "ept": "ElementsPerThread",
               "ffts_per_block": "FFTsPerBlock", "fpb": "FFTsPerBlock",
               "arch": "SM", "n": "Size", "length": "Size", "points": "Size",
               "dir": "Direction", "fft_type": "Type", "kind": "Type"}


def _find_operator(spec: dict, key: str) -> dict | None:
    k = key.strip()
    for op in spec["operators"]:
        if op["name"].lower() == k.lower():
            return op
    alias = KEY_ALIASES.get(k.lower())
    return next((o for o in spec["operators"] if o["name"] == alias), None) if alias else None


def _render_arg(param: dict, value, op: dict, errors: list[str], warnings: list[str]) -> str | None:
    name = op["name"]
    if param.get("values"):  # enum-valued
        v = str(value).split("::")[-1].strip().lower()
        match = next((x for x in param["values"] if x.lower() == v), None)
        if match is None:
            close = difflib.get_close_matches(v, param["values"], n=1)
            errors.append(f"{name}: '{value}' is not a {param['enum']}; expected one of {param['values']}"
                          + (f" (did you mean '{close[0]}'?)" if close else ""))
            return None
        return f"{param['enum']}::{match}"
    if param["kind"] == "non_type":
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, str) and value.strip().isdigit():
                value = int(value)
            else:
                errors.append(f"{name}: expected an integer, got {value!r}")
                return None
        if op["range"] and not op["range"][0] <= value <= op["range"][1]:
            warnings.append(f"{name}: {value} is outside the documented range {op['range']}")
        if op["power_of_two"] and (value < 1 or value & (value - 1)):
            warnings.append(f"{name}: {value} is not a power of two")
        if value < 0:
            errors.append(f"{name}: must be non-negative")
            return None
        return f"{value}u"
    if param["kind"] == "type":
        t = TYPE_ALIASES.get(str(value).strip().lower(), str(value).strip())
        if not re.fullmatch(r"[A-Za-z_][\w:<>, *]*", t):
            errors.append(f"{name}: '{value}' is not a valid C++ type name")
            return None
        return t
    errors.append(f"{name}: unsupported template parameter kind {param['kind']}")
    return None


def build_description(spec: dict, config: dict) -> dict:
    """Return {'expr', 'errors', 'warnings', 'operators'} for a {operator: value} mapping."""
    errors: list[str] = []
    warnings: list[str] = []
    parts: list[str] = []
    seen: set[str] = set()
    for key, value in config.items():
        op = _find_operator(spec, key)
        if op is None:
            close = difflib.get_close_matches(key, [o["name"] for o in spec["operators"]], n=2, cutoff=0.4)
            errors.append(f"unknown operator '{key}'" + (f"; did you mean {close}?" if close else
                          f"; known operators: {[o['name'] for o in spec['operators']]}"))
            continue
        if op["name"] in seen:
            errors.append(f"operator {op['name']} given twice")
            continue
        seen.add(op["name"])
        if not op["template_params"]:
            if value not in (False, None, 0):
                parts.append(f"{op['name']}()")
            continue
        if value is None:
            continue
        vals = value if isinstance(value, (list, tuple)) else [value]
        if len(vals) != len(op["template_params"]):
            errors.append(f"{op['name']}: expects {len(op['template_params'])} argument(s)")
            continue
        args = [_render_arg(p, v, op, errors, warnings) for p, v in zip(op["template_params"], vals)]
        if all(a is not None for a in args):
            parts.append(f"{op['name']}<{', '.join(args)}>()")
    for op in spec["operators"]:
        if op["required"] and op["name"] not in seen:
            warnings.append(f"required operator {op['name']} is missing")
    return {"expr": " + ".join(parts), "errors": errors, "warnings": warnings, "operators": sorted(seen)}


def with_overrides(config: dict, **over) -> dict:
    """Copy of a config where operator keys (case-insensitive) are replaced/added; None removes."""
    low = {k.lower(): k for k in config}
    out = dict(config)
    for k, v in over.items():
        out.pop(low.get(k.lower(), k), None)
        if v is not None:
            out[k] = v
    return out


def _probe_program(spec: dict, expr: str) -> tuple[str, list[dict]]:
    entry = spec["entries"][0]
    statics = [m for m in entry["members"] if m["kind"] == "static_member"]
    enums = {e["name"]: e for e in spec["enums"]}
    lines = [f"#include <{spec['library']['header']}>", "#include <cstdio>",
             f"using namespace {spec['library']['namespace']};",
             f"using Desc = decltype({expr});", f"using Fft = {entry['name']}<Desc>;",
             'static_assert(sizeof(Fft) > 0, "instantiate");', "int main() {"]
    for m in statics:
        lines.append(f'  std::printf("{m["name"]}=%lld\\n", static_cast<long long>(Fft::{m["name"]}));')
    lines += ["  return 0;", "}"]
    typed = [dict(m, enum=enums.get(re.sub(r"^const |.*::", "", m["type"]))) for m in statics]
    return "\n".join(lines) + "\n", typed


def validate_config(spec: dict, cfg: Config, config: dict) -> dict:
    built = build_description(spec, config)
    result: dict = {"valid": False, "stage": "preflight", "description": built["expr"] or None,
                    "preflight_errors": built["errors"], "warnings": built["warnings"]}
    if built["errors"] or not built["expr"]:
        if not built["expr"] and not built["errors"]:
            result["preflight_errors"] = ["empty configuration"]
        result["next_step"] = "Fix the preflight errors (see describe_operator for each operator) and call validate_config again."
        return result
    program, statics = _probe_program(spec, built["expr"])
    comp = compile_and_run(program, cfg, run=True)
    result["stage"] = "compile"
    result["valid"] = comp["ok"]
    result["compiler"] = comp["compiler"]
    if comp["ok"]:
        props: dict = {}
        for ln in (comp["run"]["stdout"] or "").splitlines():
            k, _, v = ln.partition("=")
            m = next((s for s in statics if s["name"] == k), None)
            val = int(v)
            if m and m.get("enum"):
                val = m["enum"]["values"][val]
            props[k] = val
        result["resolved_properties"] = props
        result["next_step"] = "Configuration is valid; call generate_snippet with this config."
    else:
        summary = summarize(comp, spec)
        result["errors"] = summary["errors"]
        result["retry_prompt"] = summary.get("retry_prompt")
        result["next_step"] = "Adjust the configuration according to the hints and call validate_config again."
    return result
