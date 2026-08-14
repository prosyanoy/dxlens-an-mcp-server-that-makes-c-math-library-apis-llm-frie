# Registering DxLens with MCP clients

Optional: DxLens runs standalone (`python -m dxlens.server` speaks MCP over stdio). Install first
(`pip install -e .` inside a virtualenv), then copy the matching file:

| Client | File | Copy to |
|---|---|---|
| Claude Desktop | `claude_desktop_config.json` | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\` (Windows) |
| Claude Code | `claude_code.mcp.json` | `.mcp.json` in the project root, or `claude mcp add dxlens -- python -m dxlens.server` |
| Cursor | `cursor.mcp.json` | `~/.cursor/mcp.json` or `.cursor/mcp.json` |
| VS Code (Copilot agent mode) | `vscode.mcp.json` | `.vscode/mcp.json` |

Replace `/ABSOLUTE/PATH/TO/dxlens` with your checkout. `python` must be the interpreter that has the
package installed (use the venv's `bin/python` when in doubt). Environment variables from `.env.example`
can be added under `env`. To let the agent *run* programs it writes with `check_code`, set `DXLENS_ALLOW_RUN=1`
(generated snippets from `generate_snippet` always run; they come from fixed templates).
