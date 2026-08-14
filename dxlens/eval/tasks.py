"""30 benchmark tasks with deterministic expected stdout (computed independently in Python)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

PREC = {"float": "single-precision (fp32)", "double": "double-precision (fp64)"}
SIZEOF = {"float": 4, "double": 8}
VALID_SM = {700, 750, 800, 860, 890, 900}


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    prompt: str
    expected: str
    meta: dict = field(default_factory=dict, compare=False)


def _head(n_desc: str) -> str:
    return "Using the MiniFFTDx C++ library (header minifftdx.hpp), write a complete C++17 program with a main() that " + n_desc


def _peak(n, f, t, ept=None, sm=None) -> Task:
    extra = (f" Use {ept} elements per thread." if ept else "") + (f" Target SM {sm}." if sm else "")
    p = _head(f"computes a {n}-point {PREC[t]} complex-to-complex forward FFT of x[n] = cos(2*pi*{f}*n/{n}) "
              f"(imaginary part 0) and prints exactly one line: peak_bin=<k> peak_mag=<m>, where k is the first bin whose "
              f"magnitude is the largest (use a 1e-3 tolerance) and m is its magnitude with 3 decimals.{extra}")
    return Task(f"peak-{n}-{f}-{t}" + (f"-ept{ept}" if ept else "") + (f"-sm{sm}" if sm else ""), "peak", p,
                f"peak_bin={f} peak_mag={n / 2:.3f}\n", dict(N=n, f=f, T=t, E=ept, S=sm))


def _roundtrip(n, t) -> Task:
    s = sum((i % 7) - 3 for i in range(n))
    p = _head(f"computes a {n}-point {PREC[t]} complex-to-complex forward FFT of x[n] = (n % 7) - 3 (imaginary part 0), "
              f"then the inverse FFT, divides by {n} to normalize, and prints exactly one line: sum=<s> where s is the "
              f"sum of the real parts of the result with 3 decimals.")
    return Task(f"roundtrip-{n}-{t}", "roundtrip", p, f"sum={s:.3f}\n", dict(N=n, T=t))


def _impulse(n, t) -> Task:
    p = _head(f"computes a {n}-point {PREC[t]} complex-to-complex forward FFT of the unit impulse x[0]=1 (all other "
              f"samples 0) and prints exactly one line: sum_re=<s> where s is the sum of the real parts of all {n} "
              f"output bins with 3 decimals.")
    return Task(f"impulse-{n}-{t}", "impulse", p, f"sum_re={n:.3f}\n", dict(N=n, T=t))


def _r2c(n, f, t) -> Task:
    p = _head(f"computes a {n}-point {PREC[t]} real-to-complex FFT of x[n] = sin(2*pi*{f}*n/{n}) and prints exactly one "
              f"line: bins=<count> peak_bin=<k> peak_mag=<m>, where count is the number of complex output bins the "
              f"transform produces, k the first bin with the largest magnitude (1e-3 tolerance) and m its magnitude "
              f"with 3 decimals.")
    return Task(f"r2c-{n}-{f}-{t}", "r2c", p, f"bins={n // 2 + 1} peak_bin={f} peak_mag={n / 2:.3f}\n", dict(N=n, f=f, T=t))


def _c2r(n, f, t) -> Task:
    p = _head(f"computes the {n}-point {PREC[t]} real-to-complex FFT of x[n] = cos(2*pi*{f}*n/{n}), applies the "
              f"complex-to-real inverse FFT, divides by {n} to normalize, and prints exactly one line: x1=<v>, the "
              f"recovered x[1] with 3 decimals.")
    return Task(f"c2r-{n}-{f}-{t}", "c2r", p, f"x1={math.cos(2 * math.pi * f / n):.3f}\n", dict(N=n, f=f, T=t))


def _batch(n, b, t) -> Task:
    p = _head(f"runs {b} independent {n}-point {PREC[t]} complex-to-complex forward FFTs in one batch on contiguous data, "
              f"where FFT b (b=0..{b - 1}) has input cos(2*pi*(b+1)*n/{n}); prints exactly one line: peaks=<k0>,<k1>,... "
              f"listing the first peak bin of each FFT (1e-3 tolerance).")
    return Task(f"batch-{n}-{b}-{t}", "batch", p, "peaks=" + ",".join(str(i + 1) for i in range(b)) + "\n", dict(N=n, B=b, T=t))


def _info(n, t, e, fpb) -> Task:
    p = _head(f"prints exactly one line: threads_per_fft=<t> shared_bytes=<s> for a {n}-point {PREC[t]} complex-to-complex "
              f"forward FFT configured with {e} elements per thread and {fpb} FFTs per block, where t is the number of "
              f"threads one FFT needs and s is the shared memory in bytes the library reports for one block. Query the "
              f"library's compile-time properties; do not compute them yourself.")
    return Task(f"info-{n}-{t}-ept{e}-fpb{fpb}", "info", p,
                f"threads_per_fft={n // e} shared_bytes={fpb * n * 2 * SIZEOF[t]}\n", dict(N=n, T=t, E=e, F=fpb))


def _supported(n, t, sm, fpb) -> Task:
    ok = (2 <= n <= 4096 and n & (n - 1) == 0 and sm in VALID_SM and not (t == "double" and sm < 800)
          and fpb * n * 2 * SIZEOF[t] <= 49152)
    p = _head(f"prints exactly one line: supported=<0|1> telling whether a {n}-point {PREC[t]} complex-to-complex forward "
              f"FFT targeting SM {sm} (with {fpb} FFTs per block) is a valid configuration. The program must still "
              f"compile when the configuration is invalid, so do not instantiate the FFT type directly; use the "
              f"library's non-asserting check.")
    return Task(f"supported-{n}-{t}-sm{sm}-fpb{fpb}", "supported", p, f"supported={int(ok)}\n", dict(N=n, T=t, S=sm, F=fpb))


def all_tasks() -> list[Task]:
    tasks = [
        _peak(8, 1, "float"), _peak(64, 5, "float"), _peak(256, 17, "float"), _peak(1024, 100, "float"),
        _peak(128, 9, "double"), _peak(256, 3, "float", ept=4), _peak(512, 7, "double", sm=860),
        _roundtrip(16, "float"), _roundtrip(32, "double"), _roundtrip(64, "float"),
        _roundtrip(256, "double"),
        _impulse(32, "float"), _impulse(512, "double"),
        _r2c(16, 3, "float"), _r2c(64, 10, "float"), _r2c(256, 25, "double"), _r2c(32, 1, "double"),
        _c2r(16, 2, "float"), _c2r(64, 3, "double"), _c2r(32, 1, "float"),
        _batch(16, 3, "float"), _batch(64, 4, "float"), _batch(32, 5, "double"),
        _info(256, "float", 8, 1), _info(512, "double", 4, 2), _info(64, "float", 2, 4),
        _supported(100, "float", 800, 1), _supported(64, "double", 750, 1), _supported(2048, "float", 800, 64),
        _supported(4096, "float", 860, 1),
    ]
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30
    return tasks
