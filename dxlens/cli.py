"""Command line: dxlens build | search | describe | validate | check | eval | serve."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .docs import count_tokens, render_llms_txt
from .parser import parse_header
from .service import DxLens


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "eval":  # flags belong to the harness's own parser
        from .eval.harness import main as run_eval
        return run_eval(argv[1:])
    ap = argparse.ArgumentParser(prog="dxlens", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="parse a header into spec.json + llms.txt")
    b.add_argument("header", nargs="?")
    b.add_argument("-o", "--out", default="docs/generated")
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    d = sub.add_parser("describe"); d.add_argument("name")
    v = sub.add_parser("validate"); v.add_argument("config", help='JSON, e.g. \'{"Size":256,"Type":"c2c","Direction":"forward"}\'')
    g = sub.add_parser("snippet"); g.add_argument("template"); g.add_argument("config"); g.add_argument("--params", default="{}")
    c = sub.add_parser("check", help="compile-check a C++ file"); c.add_argument("file"); c.add_argument("--run", action="store_true")
    sub.add_parser("serve", help="run the MCP server over stdio")
    sub.add_parser("eval", help="run the offline evaluation: dxlens eval --help")
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        from .server import main as serve
        serve()
        return 0
    if args.cmd == "build":
        cfg = Config.from_env()
        header = Path(args.header) if args.header else cfg.header
        spec = parse_header(header)
        text = render_llms_txt(spec)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = spec["library"]["header"].rsplit(".", 1)[0]
        (out / f"{stem}.spec.json").write_text(json.dumps(spec, indent=1) + "\n")
        (out / f"{stem}.llms.txt").write_text(text)
        print(f"wrote {out}/{stem}.spec.json and {out}/{stem}.llms.txt "
              f"({count_tokens(header.read_text())} header tokens -> {count_tokens(text)} spec tokens)")
        return 0
    lens = DxLens()
    if args.cmd == "search":
        result = lens.search_api(args.query, args.k)
    elif args.cmd == "describe":
        result = lens.describe_operator(args.name)
    elif args.cmd == "validate":
        result = lens.validate_config(json.loads(args.config))
    elif args.cmd == "snippet":
        result = lens.generate_snippet(args.template, json.loads(args.config), json.loads(args.params))
    else:
        result = lens.check_code(Path(args.file).read_text(), args.run)
    print(json.dumps(result, indent=1))
    return 0 if result.get("ok", result.get("valid", result.get("found", True))) else 1


if __name__ == "__main__":
    sys.exit(main())
