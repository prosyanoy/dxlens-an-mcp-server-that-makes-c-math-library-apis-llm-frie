import pytest

from dxlens.config import Config
from dxlens.parser import parse_header
from dxlens.service import DxLens


@pytest.fixture(scope="session")
def cfg():
    c = Config.from_env()
    if not c.cxx:
        pytest.skip("no C++ compiler available")
    return c


@pytest.fixture(scope="session")
def spec(cfg):
    return parse_header(cfg.header)


@pytest.fixture(scope="session")
def lens(cfg, spec):
    return DxLens(cfg, spec)
