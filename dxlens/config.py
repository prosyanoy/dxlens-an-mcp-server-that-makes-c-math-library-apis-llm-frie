"""Environment-driven configuration with safe local defaults (no secrets needed)."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader: KEY=VALUE lines, never overrides the real environment."""
    path = path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def find_compiler(preferred: str | None = None) -> str | None:
    for cand in ([preferred] if preferred else []) + ["g++", "clang++", "c++"]:
        if cand and shutil.which(cand):
            return cand
    return None


@dataclass
class Config:
    header: Path = REPO_ROOT / "headers" / "minifftdx.hpp"
    include_dirs: list[Path] = field(default_factory=lambda: [REPO_ROOT / "headers"])
    cxx: str | None = None
    cxxflags: list[str] = field(default_factory=lambda: ["-std=c++17", "-Wall"])
    compile_timeout: float = 60.0
    run_timeout: float = 10.0
    allow_run: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        env = os.environ
        cfg = cls()
        if env.get("DXLENS_HEADER"):
            cfg.header = Path(env["DXLENS_HEADER"]).expanduser().resolve()
        if env.get("DXLENS_INCLUDE_DIRS"):
            cfg.include_dirs = [Path(p).expanduser().resolve()
                                for p in env["DXLENS_INCLUDE_DIRS"].split(os.pathsep) if p]
        else:
            cfg.include_dirs = [cfg.header.parent]
        cfg.cxx = find_compiler(env.get("DXLENS_CXX") or None)
        if env.get("DXLENS_CXXFLAGS"):
            cfg.cxxflags = env["DXLENS_CXXFLAGS"].split()
        cfg.compile_timeout = float(env.get("DXLENS_COMPILE_TIMEOUT", cfg.compile_timeout))
        cfg.run_timeout = float(env.get("DXLENS_RUN_TIMEOUT", cfg.run_timeout))
        cfg.allow_run = env.get("DXLENS_ALLOW_RUN", "0") == "1"
        return cfg
