import json
from pathlib import Path

import jsonschema
import pytest

from dxlens.docs import render_llms_txt
from dxlens.parser import ParseError, parse_doc, parse_header

ROOT = Path(__file__).resolve().parents[1]


def test_library_metadata(spec):
    assert spec["library"]["name"] == "MiniFFTDx"
    assert spec["library"]["namespace"] == "minifftdx"
    assert spec["library"]["include"] == "#include <minifftdx.hpp>"


def test_operators_extracted_with_kinds_and_tags(spec):
    ops = {o["name"]: o for o in spec["operators"]}
    assert set(ops) == {"Size", "Precision", "Type", "Direction", "ElementsPerThread", "FFTsPerBlock", "SM", "Block"}
    size = ops["Size"]
    assert size["required"] and size["power_of_two"] and size["range"] == [2, 4096]
    assert size["template_params"][0]["kind"] == "non_type"
    assert ops["Precision"]["template_params"][0]["kind"] == "type"
    assert ops["Precision"]["default"] == "float"
    assert ops["Block"]["template_params"] == []
    assert ops["Type"]["template_params"][0]["values"] == ["c2c", "r2c", "c2r"]
    assert ops["Direction"]["template_params"][0]["values"] == ["forward", "inverse"]


def test_enums_and_combinator(spec):
    assert {e["name"]: e["values"] for e in spec["enums"]}["fft_type"] == ["c2c", "r2c", "c2r"]
    assert spec["combinators"][0]["name"] == "operator+"
    assert [t["name"] for t in spec["traits"]] == ["is_supported"]
    assert spec["functions"][0]["name"] == "reference_dft"


def test_constraints_from_static_asserts(spec):
    entry = spec["entries"][0]
    assert entry["name"] == "FFT"
    msgs = {c["message"]: c for c in entry["constraints"]}
    pow2 = msgs["Size must be a power of two in the range [2, 4096]"]
    assert pow2["operators"] == ["Size"] and "size_ok" in pow2["condition"]
    assert any(c.get("method") == "execute" for c in entry["constraints"])  # method-level guards
    size_op = next(o for o in spec["operators"] if o["name"] == "Size")
    assert any(c["id"] == pow2["id"] for c in size_op["constraints"])  # linked back to the operator


def test_members_and_examples(spec):
    entry = spec["entries"][0]
    names = [m["name"] for m in entry["members"]]
    assert {"threads_per_fft", "shared_memory_size", "execute", "execute_batch"} <= set(names)
    sigs = [m["signature"] for m in entry["members"] if m["name"] == "execute"]
    assert len(sigs) == 3 and any("real_type" in s for s in sigs)
    assert len(entry["examples"]) == 2 and "FFT<Desc>().execute(data);" in entry["examples"][0]


def test_spec_validates_against_schema(spec):
    schema = json.loads((ROOT / "schema" / "dxlens-spec.schema.json").read_text())
    jsonschema.validate(spec, schema)


def test_committed_generated_docs_are_current(spec):
    committed = json.loads((ROOT / "docs/generated/minifftdx.spec.json").read_text())
    assert committed == json.loads(json.dumps(spec)), "run `dxlens build` to refresh docs/generated"
    assert (ROOT / "docs/generated/minifftdx.llms.txt").read_text() == render_llms_txt(spec)


def test_parse_doc_tags():
    d = parse_doc("/// Hello world.\n/// @dx_operator problem\n/// @range 1 8\n/// @example\n/// A();\n/// B();\n/// @example C()")
    assert d["text"] == "Hello world."
    assert d["tags"]["range"] == ["1 8"]
    assert d["examples"] == ["A();\nB();", "C()"]


def test_missing_header_and_missing_namespace(tmp_path):
    with pytest.raises(ParseError):
        parse_header(tmp_path / "nope.hpp")
    h = tmp_path / "x.hpp"
    h.write_text("#pragma once\n/// @dx_entry\ntemplate <class T> struct Foo {};\n")
    with pytest.raises(ParseError, match="namespace"):
        parse_header(h)
