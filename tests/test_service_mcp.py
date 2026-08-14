import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


def test_describe_operator(lens):
    d = lens.describe_operator("Size<256>")
    assert d["found"] and d["kind"] == "operator" and d["range"] == [2, 4096]
    assert lens.describe_operator("fft")["kind"] == "entry"
    assert lens.describe_operator("threads_per_fft")["kind"] == "member"
    miss = lens.describe_operator("Sise")
    assert not miss["found"] and "Size" in miss["suggestions"]


def test_search_api_service(lens):
    r = lens.search_api("256 point fp32 fft", top_k=3)
    assert len(r["results"]) == 3 and r["results"][0]["score"] >= r["results"][1]["score"]


def test_check_code_run_gating(lens):
    code = "#include <cstdio>\nint main(){std::puts(\"hi\");}\n"
    r = lens.check_code(code, run=True)
    assert r["ok"] and r["run"] is None and "DXLENS_ALLOW_RUN" in r["note"]
    lens.cfg.allow_run = True
    try:
        assert lens.check_code(code, run=True)["run"]["stdout"] == "hi\n"
    finally:
        lens.cfg.allow_run = False


def test_retry_loop_end_to_end(lens):
    """An agent writes wrong code, reads the hints, fixes it, and passes."""
    bad = ("#include <minifftdx.hpp>\nint main(){ using D = decltype(cufftdx::Size<8>() + cufftdx::Type<cufftdx::fft_type::c2c>()); }\n")
    first = lens.check_code(bad)
    assert not first["ok"] and "minifftdx" in first["retry_prompt"]
    fixed = bad.replace("cufftdx", "minifftdx")
    assert lens.check_code(fixed)["ok"]


async def _mcp_session():
    params = StdioServerParameters(command=sys.executable, args=["-m", "dxlens.server"], cwd=str(ROOT))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = {t.name for t in (await s.list_tools()).tools}
            call = lambda n, a: s.call_tool(n, a)
            search = await call("search_api", {"query": "threads per fft"})
            desc = await call("describe_operator", {"name": "Size"})
            val = await call("validate_config", {"config": {"Size": 100, "Type": "c2c", "Direction": "forward"}})
            snip = await call("generate_snippet", {"template": "fft_peak", "config": {"Size": 256, "Precision": "fp32", "Type": "c2c", "Direction": "forward"}})
            res = await s.read_resource("dxlens://llms.txt")
            return tools, search, desc, val, snip, res


def _payload(result):
    return json.loads(result.content[0].text)


def test_mcp_server_over_stdio():
    tools, search, desc, val, snip, res = asyncio.run(asyncio.wait_for(_mcp_session(), 120))
    assert tools == {"search_api", "describe_operator", "validate_config", "generate_snippet", "list_templates", "check_code"}
    assert _payload(search)["results"][0]["name"] == "FFT::threads_per_fft"
    assert _payload(desc)["range"] == [2, 4096]
    v = _payload(val)
    assert v["valid"] is False and v["errors"][0]["spec_ref"]["constraint"] == "C4"
    s = _payload(snip)
    assert s["ok"] and s["compile_check"]["run"]["stdout"] == "peak_bin=1 peak_mag=128.000\n"
    assert res.contents[0].text.startswith("# MiniFFTDx")
