// MiniFFTDx: a small, header-only, CPU-executable FFT library whose API mirrors the
// style of NVIDIA cuFFTDx (compile-time "description" of a transform built by
// adding operators together). It exists as a realistic, template-heavy parsing
// target for DxLens and is fully functional on the host, so generated code can be
// compiled AND executed in tests.
//
// Doc-comment tags understood by DxLens (all optional):
//   @dx_library / @dx_operator <kind> / @dx_entry / @dx_combinator / @dx_trait
//   @required, @default <v>, @range <lo> <hi>, @pow2, @example <code...>
#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <type_traits>

/// @dx_library MiniFFTDx
/// Compile-time described FFTs. Build a description by adding operators with
/// operator+, wrap it in FFT<>, then call FFT<>::execute on host memory.
/// Everything lives in namespace minifftdx; include <minifftdx.hpp>.
namespace minifftdx {

/// Kind of transform: complex-to-complex, real-to-complex, complex-to-real.
enum class fft_type { c2c, r2c, c2r };

/// Transform direction. Inverse transforms are NOT normalized (no 1/N scaling).
enum class fft_direction { forward, inverse };

namespace detail {

struct operator_tag {};
struct description_tag {};

template <class T>
inline constexpr bool is_operator_v = std::is_base_of_v<operator_tag, T>;
template <class T>
inline constexpr bool is_description_v = std::is_base_of_v<description_tag, T>;

}  // namespace detail

/// Transform length: the number of points of one FFT.
/// @dx_operator problem
/// @required
/// @range 2 4096
/// @pow2
/// @example Size<256>()
template <unsigned int N>
struct Size : detail::operator_tag {
  static constexpr unsigned int value = N;
};

/// Floating point type of the real and imaginary parts.
/// @dx_operator problem
/// @default float
/// @example Precision<double>()
template <class T>
struct Precision : detail::operator_tag {
  using type = T;
};

/// Kind of transform (see fft_type).
/// @dx_operator problem
/// @required
/// @example Type<fft_type::c2c>()
template <fft_type T>
struct Type : detail::operator_tag {
  static constexpr fft_type value = T;
};

/// Direction of the transform (see fft_direction). Required for c2c; implied by
/// the type for r2c (forward) and c2r (inverse), where it may be omitted.
/// @dx_operator problem
/// @example Direction<fft_direction::forward>()
template <fft_direction D>
struct Direction : detail::operator_tag {
  static constexpr fft_direction value = D;
};

/// Number of elements each (virtual) thread holds. Must divide Size.
/// @dx_operator execution
/// @default min(Size, 8)
/// @range 1 4096
/// @pow2
/// @example ElementsPerThread<4>()
template <unsigned int EPT>
struct ElementsPerThread : detail::operator_tag {
  static constexpr unsigned int value = EPT;
};

/// Number of independent FFTs processed together by one execute_batch group.
/// @dx_operator execution
/// @default 1
/// @range 1 64
/// @example FFTsPerBlock<2>()
template <unsigned int FPB>
struct FFTsPerBlock : detail::operator_tag {
  static constexpr unsigned int value = FPB;
};

/// Target GPU architecture (compute capability times 100). Only validated, since
/// MiniFFTDx executes on the host.
/// @dx_operator execution
/// @default 800
/// @example SM<800>()
template <unsigned int CC>
struct SM : detail::operator_tag {
  static constexpr unsigned int value = CC;
};

/// Marker operator selecting block-level (cooperative) execution. Optional.
/// @dx_operator execution
/// @example Block()
struct Block : detail::operator_tag {};

/// An ordered list of operators; produced by adding operators together.
template <class... Ops>
struct Description : detail::description_tag {};

namespace detail {

template <class T>
struct as_description { using type = Description<T>; };
template <class... Ops>
struct as_description<Description<Ops...>> { using type = Description<Ops...>; };

template <class A, class B>
struct concat;
template <class... As, class... Bs>
struct concat<Description<As...>, Description<Bs...>> {
  using type = Description<As..., Bs...>;
};

// ---- operator lookup (first occurrence of an operator template) -----------
template <template <unsigned int> class T, class... Ops>
struct find_uint {
  static constexpr bool found = false;
  static constexpr unsigned int value = 0;
};
template <template <unsigned int> class T, unsigned int N, class... Rest>
struct find_uint<T, T<N>, Rest...> {
  static constexpr bool found = true;
  static constexpr unsigned int value = N;
};
template <template <unsigned int> class T, class H, class... Rest>
struct find_uint<T, H, Rest...> : find_uint<T, Rest...> {};

template <template <fft_type> class T, class... Ops>
struct find_type {
  static constexpr bool found = false;
  static constexpr fft_type value = fft_type::c2c;
};
template <template <fft_type> class T, fft_type V, class... Rest>
struct find_type<T, T<V>, Rest...> {
  static constexpr bool found = true;
  static constexpr fft_type value = V;
};
template <template <fft_type> class T, class H, class... Rest>
struct find_type<T, H, Rest...> : find_type<T, Rest...> {};

template <template <fft_direction> class T, class... Ops>
struct find_dir {
  static constexpr bool found = false;
  static constexpr fft_direction value = fft_direction::forward;
};
template <template <fft_direction> class T, fft_direction V, class... Rest>
struct find_dir<T, T<V>, Rest...> {
  static constexpr bool found = true;
  static constexpr fft_direction value = V;
};
template <template <fft_direction> class T, class H, class... Rest>
struct find_dir<T, H, Rest...> : find_dir<T, Rest...> {};

template <class... Ops>
struct find_precision {
  static constexpr bool found = false;
  using type = float;
};
template <class P, class... Rest>
struct find_precision<Precision<P>, Rest...> {
  static constexpr bool found = true;
  using type = P;
};
template <class H, class... Rest>
struct find_precision<H, Rest...> : find_precision<Rest...> {};

// ---- everything MiniFFTDx knows about a description -----------------------
template <class Desc>
struct traits;
template <class... Ops>
struct traits<Description<Ops...>> {
  using size_op = find_uint<Size, Ops...>;
  using ept_op = find_uint<ElementsPerThread, Ops...>;
  using fpb_op = find_uint<FFTsPerBlock, Ops...>;
  using sm_op = find_uint<SM, Ops...>;
  using type_op = find_type<Type, Ops...>;
  using dir_op = find_dir<Direction, Ops...>;
  using prec_op = find_precision<Ops...>;

  static constexpr bool has_size = size_op::found;
  static constexpr bool has_type = type_op::found;
  static constexpr bool has_direction = dir_op::found;

  static constexpr unsigned int size = size_op::value;
  static constexpr fft_type type = type_op::value;
  static constexpr unsigned int sm = sm_op::found ? sm_op::value : 800u;
  static constexpr unsigned int fpb = fpb_op::found ? fpb_op::value : 1u;
  static constexpr unsigned int ept =
      ept_op::found ? ept_op::value : (size < 8u && size != 0u ? size : 8u);
  using precision = typename prec_op::type;
  // c2c: as given. r2c: forward. c2r: inverse.
  static constexpr fft_direction direction =
      type == fft_type::r2c   ? fft_direction::forward
      : type == fft_type::c2r ? fft_direction::inverse
                              : dir_op::value;

  // Individual validity checks, shared by FFT<> static_asserts and is_supported.
  static constexpr bool size_ok =
      has_size && size >= 2 && size <= 4096 && (size & (size - 1)) == 0;
  static constexpr bool precision_ok =
      std::is_same_v<precision, float> || std::is_same_v<precision, double>;
  static constexpr bool direction_ok =
      type != fft_type::c2c || has_direction;
  static constexpr bool direction_consistent =
      !has_direction || type == fft_type::c2c ||
      (type == fft_type::r2c && dir_op::value == fft_direction::forward) ||
      (type == fft_type::c2r && dir_op::value == fft_direction::inverse);
  static constexpr bool ept_ok = size_ok ? (ept >= 1 && size % ept == 0) : true;
  static constexpr bool fpb_ok = fpb >= 1 && fpb <= 64;
  static constexpr bool sm_ok = sm == 700 || sm == 750 || sm == 800 ||
                                sm == 860 || sm == 890 || sm == 900;
  static constexpr bool double_ok =
      !std::is_same_v<precision, double> || sm >= 800;
  static constexpr unsigned long long shared_bytes =
      static_cast<unsigned long long>(fpb) * size * 2ull * sizeof(precision);
  static constexpr bool shared_ok = shared_bytes <= 49152ull;

  static constexpr bool supported =
      has_size && has_type && size_ok && precision_ok && direction_ok &&
      direction_consistent && ept_ok && fpb_ok && sm_ok && double_ok &&
      shared_ok;
};

template <class P>
void fft_inplace(std::complex<P>* x, unsigned int n, bool inverse) {
  for (unsigned int i = 1, j = 0; i < n; ++i) {  // bit reversal
    unsigned int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(x[i], x[j]);
  }
  const double sign = inverse ? 1.0 : -1.0;
  for (unsigned int len = 2; len <= n; len <<= 1) {
    const double ang = sign * 2.0 * 3.14159265358979323846 / len;
    for (unsigned int i = 0; i < n; i += len) {
      for (unsigned int k = 0; k < len / 2; ++k) {
        const std::complex<P> w(static_cast<P>(std::cos(ang * k)),
                                static_cast<P>(std::sin(ang * k)));
        const std::complex<P> u = x[i + k];
        const std::complex<P> v = x[i + k + len / 2] * w;
        x[i + k] = u + v;
        x[i + k + len / 2] = u - v;
      }
    }
  }
}

}  // namespace detail

/// Combine two operators and/or descriptions into one Description. This is the
/// only way to build a description: Size<64>() + Type<fft_type::c2c>() + ...
/// @dx_combinator
template <class A, class B,
          class = std::enable_if_t<(detail::is_operator_v<A> || detail::is_description_v<A>) &&
                                   (detail::is_operator_v<B> || detail::is_description_v<B>)>>
constexpr auto operator+(const A&, const B&) {
  return typename detail::concat<typename detail::as_description<A>::type,
                                 typename detail::as_description<B>::type>::type{};
}

/// Non-asserting check: true when Desc is a complete, valid FFT description.
/// Use it to test a configuration without triggering a compile error.
/// @dx_trait
/// @example static_assert(is_supported<decltype(Size<64>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>())>::value);
template <class Desc, class = void>
struct is_supported : std::false_type {};
template <class Desc>
struct is_supported<Desc, std::enable_if_t<detail::is_description_v<Desc>>>
    : std::integral_constant<bool, detail::traits<Desc>::supported> {};

/// A host-executable FFT described by Desc (a Description built with operator+).
/// Constraints are enforced with static_assert when FFT<Desc> is instantiated.
/// Execution is in place for c2c, out of place for r2c and c2r. Inverse
/// transforms are not normalized. r2c produces size/2+1 complex outputs; c2r
/// consumes size/2+1 complex inputs and writes size real outputs.
/// @dx_entry
/// @example
/// using Desc = decltype(Size<8>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>());
/// std::complex<float> data[8] = {};
/// FFT<Desc>().execute(data);
/// @example
/// using Desc = decltype(Size<16>() + Precision<double>() + Type<fft_type::r2c>());
/// double in[16] = {}; std::complex<double> out[9];
/// FFT<Desc>().execute(in, out);
template <class Desc>
struct FFT {
  using description = Desc;
  using traits = detail::traits<Desc>;

  static_assert(detail::is_description_v<Desc>,
                "FFT requires a Description built by adding operators with operator+");
  static_assert(traits::has_size, "Size<N> operator is required");
  static_assert(traits::has_type, "Type<fft_type> operator is required");
  static_assert(!traits::has_size || traits::size_ok,
                "Size must be a power of two in the range [2, 4096]");
  static_assert(traits::precision_ok, "Precision must be float or double");
  static_assert(!traits::has_type || traits::direction_ok,
                "Direction<> is required for c2c transforms");
  static_assert(traits::direction_consistent,
                "Direction conflicts with Type: r2c is forward-only and c2r is inverse-only");
  static_assert(traits::ept_ok, "ElementsPerThread must evenly divide Size");
  static_assert(traits::fpb_ok, "FFTsPerBlock must be in the range [1, 64]");
  static_assert(traits::sm_ok, "SM must be one of 700, 750, 800, 860, 890, 900");
  static_assert(traits::double_ok, "double precision requires SM >= 800");
  static_assert(traits::shared_ok,
                "FFTsPerBlock * Size * sizeof(complex) must not exceed 48 KiB of shared memory");

  using value_type = std::complex<typename traits::precision>;
  using real_type = typename traits::precision;

  /// Number of points of one FFT.
  static constexpr unsigned int size = traits::size;
  static constexpr fft_type type = traits::type;
  static constexpr fft_direction direction = traits::direction;
  /// Elements held per thread (default min(Size, 8)).
  static constexpr unsigned int elements_per_thread = traits::ept;
  /// Number of FFTs per execute_batch group (default 1).
  static constexpr unsigned int ffts_per_block = traits::fpb;
  /// Target architecture (default 800).
  static constexpr unsigned int sm = traits::sm;
  /// Threads needed for one FFT: Size / ElementsPerThread.
  static constexpr unsigned int threads_per_fft = traits::size / traits::ept;
  /// Per-thread register storage in elements (== elements_per_thread).
  static constexpr unsigned int storage_size = traits::ept;
  /// Shared memory bytes needed by one group: FFTsPerBlock * Size * sizeof(complex).
  static constexpr unsigned int shared_memory_size =
      static_cast<unsigned int>(traits::shared_bytes);
  /// Number of input elements per FFT (c2c: size, r2c: size reals, c2r: size/2+1).
  static constexpr unsigned int input_length =
      traits::type == fft_type::c2r ? traits::size / 2 + 1 : traits::size;
  /// Number of output elements per FFT (c2c: size, r2c: size/2+1, c2r: size reals).
  static constexpr unsigned int output_length =
      traits::type == fft_type::r2c ? traits::size / 2 + 1 : traits::size;

  /// In-place c2c transform of `size` elements. Only valid for Type<fft_type::c2c>.
  void execute(value_type* data) const {
    static_assert(traits::type == fft_type::c2c,
                  "execute(data) is only available for c2c transforms; use execute(in, out) for r2c/c2r");
    detail::fft_inplace(data, traits::size, traits::direction == fft_direction::inverse);
  }

  /// r2c: `size` real inputs -> size/2+1 complex outputs. Only valid for Type<fft_type::r2c>.
  void execute(const real_type* in, value_type* out) const {
    static_assert(traits::type == fft_type::r2c,
                  "execute(const real*, complex*) is only available for r2c transforms");
    value_type buf[traits::size];
    for (unsigned int i = 0; i < traits::size; ++i) buf[i] = value_type(in[i], 0);
    detail::fft_inplace(buf, traits::size, false);
    for (unsigned int i = 0; i < traits::size / 2 + 1; ++i) out[i] = buf[i];
  }

  /// c2r: size/2+1 complex inputs (Hermitian half spectrum) -> `size` real outputs,
  /// unnormalized. Only valid for Type<fft_type::c2r>.
  void execute(const value_type* in, real_type* out) const {
    static_assert(traits::type == fft_type::c2r,
                  "execute(const complex*, real*) is only available for c2r transforms");
    value_type buf[traits::size];
    for (unsigned int i = 0; i < traits::size / 2 + 1; ++i) buf[i] = in[i];
    for (unsigned int i = traits::size / 2 + 1; i < traits::size; ++i)
      buf[i] = std::conj(in[traits::size - i]);
    detail::fft_inplace(buf, traits::size, true);
    for (unsigned int i = 0; i < traits::size; ++i) out[i] = buf[i].real();
  }

  /// c2c transform of `batch` contiguous FFTs (batch * size elements), in place.
  void execute_batch(value_type* data, unsigned int batch) const {
    static_assert(traits::type == fft_type::c2c,
                  "execute_batch is only available for c2c transforms");
    for (unsigned int b = 0; b < batch; ++b) execute(data + static_cast<std::size_t>(b) * traits::size);
  }
};

/// Naive O(N^2) reference DFT, useful to check results. Direction as in fft_direction;
/// inverse is unnormalized.
/// @dx_function
/// @example reference_dft(in, out, 8, fft_direction::forward);
template <class P>
void reference_dft(const std::complex<P>* in, std::complex<P>* out, unsigned int n,
                   fft_direction dir) {
  const double sign = dir == fft_direction::forward ? -1.0 : 1.0;
  for (unsigned int k = 0; k < n; ++k) {
    std::complex<double> acc(0, 0);
    for (unsigned int t = 0; t < n; ++t) {
      const double ang = sign * 2.0 * 3.14159265358979323846 * k * t / n;
      acc += std::complex<double>(in[t]) * std::complex<double>(std::cos(ang), std::sin(ang));
    }
    out[k] = std::complex<P>(static_cast<P>(acc.real()), static_cast<P>(acc.imag()));
  }
}

}  // namespace minifftdx
