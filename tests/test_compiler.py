from dxlens.compiler import add_hints, compile_and_run, parse_diagnostics, summarize

GCC = """main.cpp: In function 'int main()':
main.cpp:4:3: error: 'Foo' was not declared in this scope
    4 |   Foo f;
      |   ^~~
In file included from main.cpp:1:
h.hpp: In instantiation of 'struct FFT<X>':
main.cpp:6:20:   required from here
h.hpp:313:35: error: static assertion failed: Size must be a power of two in the range [2, 4096]
  313 |   static_assert(a,
h.hpp:9:1: note: candidate: 'void f()'
"""
CLANG = """main.cpp:4:3: error: use of undeclared identifier 'Foo'
  Foo f;
  ^
main.cpp:5:1: warning: unused variable
"""


def test_parse_gcc_diagnostics_trace_and_notes():
    d = parse_diagnostics(GCC)
    assert [x["line"] for x in d] == [4, 313]
    assert d[0]["code"] == "Foo f;" and d[0]["where"] == "user"
    assert d[1]["where"] == "library" and d[1]["user_line"] == 6
    assert d[1]["notes"] == ["h.hpp:9: candidate: 'void f()'"]


def test_parse_clang_diagnostics():
    d = parse_diagnostics(CLANG)
    assert d[0]["message"] == "use of undeclared identifier 'Foo'" and d[0]["code"] == "Foo f;"
    assert d[1]["severity"] == "warning"


def test_hints_for_namespace_and_static_assert(spec):
    d = add_hints(parse_diagnostics("main.cpp:3:1: error: 'cufftdx' has not been declared\n"), spec)
    assert "minifftdx" in d[0]["hint"]
    d = add_hints(parse_diagnostics("main.cpp:3:1: error: 'Sise' was not declared in this scope\n"), spec)
    assert "Size" in d[0]["hint"]
    d = add_hints(parse_diagnostics("h.hpp:3:1: error: static assertion failed: ElementsPerThread must evenly divide Size\n"), spec)
    assert d[0]["spec_ref"]["constraint"] == "C8" and "ElementsPerThread" in d[0]["hint"]
    d = add_hints(parse_diagnostics("main.cpp:3:1: error: minifftdx.hpp: No such file or directory\n"), spec)
    assert "#include <minifftdx.hpp>" in d[0]["hint"]


GOOD = """#include <minifftdx.hpp>
#include <cstdio>
using namespace minifftdx;
int main() { using F = FFT<decltype(Size<8>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>())>;
  std::printf("%u\\n", F::size); }
"""


def test_compile_and_run_success(cfg):
    r = compile_and_run(GOOD, cfg, run=True)
    assert r["ok"] and r["run"]["stdout"] == "8\n"


def test_compile_failure_gives_structured_retry_info(cfg, spec):
    bad = GOOD.replace("Size<8>", "Size<9>")
    s = summarize(compile_and_run(bad, cfg), spec)
    assert not s["ok"] and s["errors"][0]["spec_ref"]["constraint"] == "C4"
    assert s["errors"][0]["user_line"] == 5  # where F::size instantiates the type
    assert "retry_prompt" in s and "C4" in s["retry_prompt"]


def test_runtime_failure_and_timeout(cfg):
    import dataclasses
    r = compile_and_run("int main(){return 3;}", cfg, run=True)
    assert not r["ok"] and r["run"]["returncode"] == 3
    quick = dataclasses.replace(cfg, run_timeout=0.5)
    r = compile_and_run("int main(){for(;;){}}", quick, run=True)
    assert r["run"]["timed_out"] and not r["ok"]


def test_missing_compiler():
    import dataclasses

    from dxlens.config import Config
    r = compile_and_run("int main(){}", dataclasses.replace(Config(), cxx=None))
    assert not r["ok"] and r["stage"] == "setup"
