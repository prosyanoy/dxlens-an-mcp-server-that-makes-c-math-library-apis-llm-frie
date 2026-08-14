"""MCP server (stdio by default). Tools are thin wrappers over dxlens.service.DxLens."""
from __future__ import annotations

import json
import sys

from mcp.server.fastmcp import FastMCP

from .service import DxLens

INSTRUCTIONS = """DxLens exposes a C++ template math library as a compact spec plus a compile-check loop.
Typical flow: search_api -> describe_operator -> validate_config -> generate_snippet (compiles and runs it).
When you write your own code, pass it to check_code and use the structured errors/hints to retry."""


def create_server(lens: DxLens | None = None) -> FastMCP:
    lens = lens or DxLens()
    mcp = FastMCP("dxlens", instructions=INSTRUCTIONS)

    @mcp.tool()
    def search_api(query: str, top_k: int = 5) -> dict:
        """Search the library API (operators, members, constraints, helpers) with BM25 and return ranked hits."""
        return lens.search_api(query, top_k)

    @mcp.tool()
    def describe_operator(name: str) -> dict:
        """Full spec record for an operator (e.g. 'Size'), entry (e.g. 'FFT'), enum, trait or member: parameters, domain, defaults, constraints, examples."""
        return lens.describe_operator(name)

    @mcp.tool()
    def validate_config(config: dict) -> dict:
        """Check a configuration such as {"Size": 256, "Precision": "float", "Type": "c2c", "Direction": "forward"}.
        Maps it to operators, then compiles it so the library's own compile-time constraints decide validity.
        Returns resolved properties when valid, otherwise structured errors with hints."""
        return lens.validate_config(config)

    @mcp.tool()
    def list_templates() -> dict:
        """List snippet templates usable with generate_snippet."""
        return lens.list_templates()

    @mcp.tool()
    def generate_snippet(template: str, config: dict, params: dict | None = None, run: bool = True) -> dict:
        """Render a template for a configuration, compile it with the local compiler (and run it) and return the code
        plus compile_check diagnostics usable for retries."""
        return lens.generate_snippet(template, config, params, run)

    @mcp.tool()
    def check_code(code: str, run: bool = False) -> dict:
        """Compile-check a complete C++ program you wrote (main.cpp). Returns structured diagnostics with hints
        derived from the API spec and a ready-made retry_prompt. run=true only works if the server enables DXLENS_ALLOW_RUN=1."""
        return lens.check_code(code, run)

    @mcp.resource("dxlens://llms.txt")
    def llms_txt() -> str:
        """Compact llms.txt-style documentation of the whole API."""
        return lens.llms_txt

    @mcp.resource("dxlens://spec.json")
    def spec_json() -> str:
        """The full machine-readable API spec."""
        return json.dumps(lens.spec, indent=1)

    return mcp


def main() -> None:
    try:
        server = create_server()
    except Exception as exc:  # surface startup problems on stderr; stdout is the protocol channel
        print(f"dxlens: failed to start: {exc}", file=sys.stderr)
        raise SystemExit(1)
    server.run()


if __name__ == "__main__":
    main()
