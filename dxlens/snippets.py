"""generate_snippet: render a code template for a validated configuration."""
from __future__ import annotations

from pathlib import Path
from string import Template

from .validate import build_description, with_overrides

TEMPLATE_DIR = Path(__file__).parent / "templates"

# name -> metadata. `types` lists the Type<> values the template supports (None = any).
TEMPLATES: dict[str, dict] = {
    "fft_peak": {"summary": "c2c FFT of cos(2*pi*f*n/N); prints peak_bin and peak_mag.",
                 "types": ["c2c"], "params": {"frequency": 1}},
    "roundtrip": {"summary": "c2c forward then inverse FFT, normalised by 1/N; prints sum and error check.",
                  "types": ["c2c"], "params": {"tolerance": "1e-3"}},
    "impulse": {"summary": "c2c FFT of a unit impulse; prints the sum of real parts (= N).",
                "types": ["c2c"], "params": {}},
    "r2c_peak": {"summary": "r2c FFT of sin(2*pi*f*n/N); prints bin count and the peak.",
                 "types": ["r2c"], "params": {"frequency": 1}},
    "c2r_roundtrip": {"summary": "r2c then c2r of cos(2*pi*f*n/N), normalised; prints x[1].",
                      "types": ["r2c", "c2r"], "params": {"frequency": 1}},
    "batched": {"summary": "execute_batch over several c2c FFTs; prints the peak bin of each.",
                "types": ["c2c"], "params": {"batch": 4}},
    "properties": {"summary": "Prints every compile-time property of the FFT type.", "types": None, "params": {}},
}


def list_templates() -> list[dict]:
    return [{"name": n, "summary": m["summary"], "supported_types": m["types"] or "any",
             "params": m["params"]} for n, m in TEMPLATES.items()]


def _type_of(spec: dict, config: dict) -> str | None:
    for k, v in config.items():
        if k.lower() in ("type", "fft_type", "kind"):
            return str(v).split("::")[-1].lower()
    return None


def render(spec: dict, template: str, config: dict, params: dict | None = None) -> dict:
    """Return {'code'} or {'errors'}; never raises on bad user input."""
    if template not in TEMPLATES:
        return {"errors": [f"unknown template '{template}'; available: {sorted(TEMPLATES)}"]}
    meta = TEMPLATES[template]
    params = {**meta["params"], **(params or {})}
    unknown = set(params) - set(meta["params"])
    if unknown:
        return {"errors": [f"unknown params {sorted(unknown)} for '{template}'; allowed: {sorted(meta['params'])}"]}
    ftype = _type_of(spec, config)
    if meta["types"] and ftype not in meta["types"]:
        return {"errors": [f"template '{template}' needs Type in {meta['types']}, got {ftype!r}"]}

    built = build_description(spec, config)
    if built["errors"] or not built["expr"]:
        return {"errors": built["errors"] or ["empty configuration"]}
    lib, entry = spec["library"], spec["entries"][0]
    subs = {"HEADER": lib["header"], "NS": lib["namespace"], "ENTRY": entry["name"], "DESC": built["expr"]}
    for k, v in params.items():
        if k in ("frequency", "batch"):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                return {"errors": [f"param '{k}' must be a positive integer"]}
        subs[k.upper()] = str(v)
    if template == "roundtrip":
        subs["DESC_FWD"] = build_description(spec, with_overrides(config, Direction="forward"))["expr"]
        subs["DESC_INV"] = build_description(spec, with_overrides(config, Direction="inverse"))["expr"]
    if template == "c2r_roundtrip":
        subs["DESC_R2C"] = build_description(spec, with_overrides(config, Type="r2c", Direction=None))["expr"]
        subs["DESC_C2R"] = build_description(spec, with_overrides(config, Type="c2r", Direction=None))["expr"]
    if template == "properties":
        statics = [m["name"] for m in entry["members"] if m["kind"] == "static_member"]
        subs["PROPERTY_LINES"] = "\n".join(
            f'  std::printf("{n}=%lld\\n", static_cast<long long>(Fft::{n}));' for n in statics)
    text = (TEMPLATE_DIR / f"{template}.cpp.tmpl").read_text()
    return {"code": Template(text).substitute(subs), "description": built["expr"], "warnings": built["warnings"]}
