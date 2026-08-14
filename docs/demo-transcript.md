# Demo transcript: agent asks for "a 256-point fp32 FFT"

Real tool output from `DxLens` (captured from the tools in-process; compiler: g++).

### 1. search_api("256 point fp32 fft", top_k=3)
```json
{
 "query": "256 point fp32 fft",
 "results": [
  {
   "kind": "operator",
   "name": "Precision",
   "score": 6.918,
   "summary": "Precision operator Precision<T>() problem Floating point type of the real and imaginary parts. Precision<double>() optional default float Precision must be float or double",
   "ref": {
    "operator": "Precision"
   }
  },
  {
   "kind": "constraint",
   "name": "C5",
   "score": 3.441,
   "summary": "constraint C5 Precision must be float or double Precision",
   "ref": {
    "entry": "FFT",
    "constraint": "C5"
   }
  },
  {
   "kind": "operator",
   "name": "Size",
   "score": 2.828,
   "summary": "Size operator Size<N>() problem Transform length: the number of points of one FFT. Size<256>() required default  Size<N> operator is required Size must be a power of two in the range [2, 4096] ElementsPerThread must evenly divide Size FFTsP",
   "ref": {
    "operator": "Size"
   }
  }
 ],
 "tip": "Call describe_operator(name) for the full record of a hit."
}
```

### 2. validate_config({"Size":256,"Precision":"fp32","Type":"c2c","Direction":"forward"})
```json
{
 "valid": true,
 "stage": "compile",
 "description": "Size<256u>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>()",
 "preflight_errors": [],
 "warnings": [],
 "compiler": "g++",
 "resolved_properties": {
  "size": 256,
  "type": "c2c",
  "direction": "forward",
  "elements_per_thread": 8,
  "ffts_per_block": 1,
  "sm": 800,
  "threads_per_fft": 32,
  "storage_size": 8,
  "shared_memory_size": 2048,
  "input_length": 256,
  "output_length": 256
 },
 "next_step": "Configuration is valid; call generate_snippet with this config."
}
```

### 3. Agent's first hand-written attempt (wrong namespace) via check_code
```cpp
#include <minifftdx.hpp>
int main(){
  using D = decltype(cufftdx::Size<256>() + cufftdx::Type<cufftdx::fft_type::c2c>());
}
```
```
Compilation failed. Fix these and call again:
- line 3: 'cufftdx' was not declared in this scope  HINT: The namespace of this library is `minifftdx` (not `cufftdx`); use `#include <minifftdx.hpp>` and `using namespace minifftdx;`.
- line 3: 'cufftdx' is not a class, namespace, or enumeration
- line 3: expected primary-expression before ')' token
- line 3: 'cufftdx' is not a class, namespace, or enumeration
- line 3: 'cufftdx' is not a class, namespace, or enumeration
```

### 4. generate_snippet("fft_peak", ...) compiles and runs
```cpp
#include <cstdio>
#include <cmath>
#include <minifftdx.hpp>
using namespace minifftdx;

int main() {
  using Fft = FFT<decltype(Size<256u>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>())>;
  using real_t = typename Fft::real_type;
  constexpr unsigned N = Fft::size;
  typename Fft::value_type data[N];
  for (unsigned n = 0; n < N; ++n)
    data[n] = typename Fft::value_type(static_cast<real_t>(std::cos(2.0 * 3.14159265358979323846 * 5 * n / N)), 0);
  Fft().execute(data);
  unsigned peak = 0;
  double best = std::abs(data[0]);
  for (unsigned k = 1; k < N; ++k) {
    const double mag = std::abs(data[k]);
    if (mag > best + 1e-3) { best = mag; peak = k; }
  }
  std::printf("peak_bin=%u peak_mag=%.3f\n", peak, best);
  return 0;
}
```
Program output: `peak_bin=5 peak_mag=128.000`
