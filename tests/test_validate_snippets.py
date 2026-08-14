import pytest

from dxlens import snippets
from dxlens.validate import build_description, validate_config, with_overrides

OK = {"Size": 256, "Precision": "fp32", "Type": "c2c", "Direction": "forward"}


def test_build_description_aliases_and_enums(spec):
    b = build_description(spec, {"size": 64, "precision": "fp64", "type": "fft_type::r2c", "ept": 4, "Block": True})
    assert b["errors"] == []
    assert b["expr"] == "Size<64u>() + Precision<double>() + Type<fft_type::r2c>() + ElementsPerThread<4u>() + Block()"


@pytest.mark.parametrize("cfg_,frag", [({"Sise": 8}, "did you mean"), ({"Type": "c3c"}, "expected one of"),
                                       ({"Size": "big"}, "expected an integer")])
def test_preflight_errors(spec, cfg, cfg_, frag):
    r = validate_config(spec, cfg, cfg_)
    assert not r["valid"] and r["stage"] == "preflight" and frag in " ".join(r["preflight_errors"])


def test_valid_config_reports_resolved_properties(spec, cfg):
    r = validate_config(spec, cfg, OK)
    assert r["valid"], r
    p = r["resolved_properties"]
    assert (p["size"], p["type"], p["direction"], p["threads_per_fft"], p["shared_memory_size"]) == (256, "c2c", "forward", 32, 2048)


@pytest.mark.parametrize("override,constraint", [
    ({"Size": 100}, "C4"), ({"Direction": None}, "C6"), ({"ElementsPerThread": 16, "Size": 8}, "C8"),
    ({"Precision": "double", "SM": 750}, "C11"), ({"SM": 123}, "C10"), ({"Size": 4096, "FFTsPerBlock": 2, "Precision": "double"}, "C12"),
    ({"Precision": "int"}, "C5"), ({"Type": "r2c", "Direction": "inverse"}, "C7")])
def test_invalid_configs_map_to_library_constraints(spec, cfg, override, constraint):
    r = validate_config(spec, cfg, with_overrides(OK, **override))
    assert not r["valid"], r
    assert constraint in {e["spec_ref"]["constraint"] for e in r["errors"] if e.get("spec_ref") and "constraint" in e["spec_ref"]}


def test_domain_warnings_agree_with_compiler(spec, cfg):
    for size in (1, 3, 100, 8192, 2, 4096):
        c = with_overrides(OK, Size=size, Precision="float")
        warned = bool(build_description(spec, c)["warnings"])
        assert warned == (not validate_config(spec, cfg, c)["valid"]), size


def test_required_operator_warning(spec):
    assert any("Size" in w for w in build_description(spec, {"Type": "c2c"})["warnings"])


EXPECTED = [
    ("fft_peak", {"Size": 64, "Type": "c2c", "Direction": "forward"}, {"frequency": 5}, "peak_bin=5 peak_mag=32.000\n"),
    ("roundtrip", {"Size": 32, "Precision": "double", "Type": "c2c", "Direction": "forward"}, {}, "sum=-6.000 max_error_ok=1\n"),
    ("impulse", {"Size": 16, "Type": "c2c", "Direction": "forward"}, {}, "sum_re=16.000\n"),
    ("r2c_peak", {"Size": 16, "Type": "r2c"}, {"frequency": 3}, "bins=9 peak_bin=3 peak_mag=8.000\n"),
    ("c2r_roundtrip", {"Size": 16, "Type": "r2c"}, {"frequency": 2}, "x1=0.707\n"),
    ("batched", {"Size": 16, "Type": "c2c", "Direction": "forward"}, {"batch": 3}, "peaks=1,2,3\n"),
]


@pytest.mark.parametrize("name,config,params,out", EXPECTED)
def test_templates_compile_run_and_are_correct(lens, name, config, params, out):
    r = lens.generate_snippet(name, config, params)
    assert r["ok"], r.get("compile_check")
    assert r["compile_check"]["run"]["stdout"] == out


def test_properties_template(lens):
    r = lens.generate_snippet("properties", {"Size": 8, "Type": "r2c"})
    assert r["ok"] and "output_length=5" in r["compile_check"]["run"]["stdout"]


def test_generate_snippet_rejects_bad_requests(lens):
    assert "needs Type" in lens.generate_snippet("fft_peak", {"Size": 8, "Type": "r2c"})["errors"][0]
    assert "unknown template" in lens.generate_snippet("nope", OK)["errors"][0]
    assert "unknown params" in lens.generate_snippet("impulse", OK, {"bogus": 1})["errors"][0]
    assert "positive integer" in lens.generate_snippet("fft_peak", OK, {"frequency": -2})["errors"][0]


def test_generate_snippet_surfaces_compile_diagnostics(lens):
    r = lens.generate_snippet("fft_peak", {"Size": 100, "Type": "c2c", "Direction": "forward"})
    assert not r["ok"] and r["compile_check"]["errors"][0]["spec_ref"]["constraint"] == "C4"
    assert r["code"] and "next_step" in r


def test_template_listing():
    assert {t["name"] for t in snippets.list_templates()} >= {"fft_peak", "roundtrip", "batched"}
