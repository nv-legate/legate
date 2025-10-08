/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/cuda/detail/cuda_driver_api.h>

#include <legate_defines.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <optional>
#include <regex>
#include <stdexcept>
#include <type_traits>

#if __has_include(<cuda.h>) || LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <cuda.h>

#define LEGATE_CUDA_VERSION CUDA_VERSION
#define CHECK_ABI_COMPATIBLE(cu_function, our_function)                      \
  static_assert(abi_detail::check_abi_compatible(cu_function, our_function), \
                LEGATE_STRINGIZE(cu_function) " and " LEGATE_STRINGIZE(      \
                  our_function) " are not ABI compatible")
#else
#define LEGATE_CUDA_VERSION 9999999  // dummy value
#define CU_GET_PROC_ADDRESS_DEFAULT 0
#define CHECK_ABI_COMPATIBLE(cu_function, our_function) static_assert(true)
#endif

namespace abi_detail {

namespace {

template <typename T1, typename T2, typename = void>
struct abi_compatible : std::false_type {};  // NOLINT(readability-identifier-naming)

// Two types that are the same are always ABI compatible
template <typename T>
struct abi_compatible<T, T> : std::true_type {};

// Specialization when either T1 or T2 is an enum. In this case, all that matters is that both
// are the same size
template <typename T1, typename T2>
struct abi_compatible<T1, T2, std::enable_if_t<std::is_enum_v<T1> || std::is_enum_v<T2>>>
  : std::conditional_t<sizeof(T1) == sizeof(T2), std::true_type, std::false_type> {};

// The enable_if_t is needed because without it
//
// abi_compatible<Type **, Type **>
//
// Is an ambiguous overload. It is either abi_compatible<T, T> (with T = Type **), or
// abi_compatible<T1*, T2*> (where T1 = T2 = Type *). We still need to have this overload in that
// case, because we still want to catch instances where we spoof enums with their underlying values:
//
// enum TheRealType { ... }; (underlying type is int)
// using OurType : int;
//
// In this case, a function taking OurType, or OurType * is ABI compatible with the real enum,
// but won't necessarily have the exact same pointed-to type.
template <typename T1, typename T2>
struct abi_compatible<T1*, T2*, std::enable_if_t<!std::is_same_v<T1, T2>>>
  : abi_compatible<T1, T2> {};

template <typename T1, typename T2>
inline constexpr bool abi_compatible_v = abi_compatible<T1, T2>::value;

// NOLINTBEGIN
enum Foo_enum {};

using FooEnum = int;

struct Foo_st;
using FooStruct = Foo_st*;
// NOLINTEND

static_assert(abi_compatible_v<int, int>);
static_assert(!abi_compatible_v<int, float>);
static_assert(abi_compatible_v<const char**, const char**>);
static_assert(!abi_compatible_v<const char*, const int*>);
static_assert(abi_compatible_v<FooEnum, Foo_enum>);
static_assert(abi_compatible_v<FooEnum*, Foo_enum*>);
static_assert(abi_compatible_v<Foo_st*, FooStruct>);

// ==========================================================================================

template <typename R1, typename... Args1, typename R2, typename... Args2>
[[nodiscard]] constexpr bool check_abi_compatible(R1 (*)(Args1...), R2 (*)(Args2...))
{
  if constexpr (abi_compatible_v<R1, R2> && (sizeof...(Args1) == sizeof...(Args2))) {
    return std::conjunction_v<abi_compatible<Args1, Args2>...>;
  }
  return false;
}

}  // namespace

}  // namespace abi_detail

namespace legate::cuda::detail {

namespace {

[[noreturn]] void throw_null_driver_request(std::string_view fn_name)
{
  throw legate::detail::TracedException<std::runtime_error>{fmt::format(
    "cuGetProcAddress returned null {} function pointers for CUDA 12.x and 13.x. "
    "Please validate your CUDA driver has these symbols and are load-able via cuGetProcAddress.",
    fn_name)};
}

[[noreturn]] void abort_null_driver_fn(std::string_view fn_name)
{
  LEGATE_ABORT(fmt::format("{} function pointers "
                           "are null but expected to be non-null. Driver loading "
                           "checks at least one function pointer is non-null."
                           "This indicates checks are incorrect or "
                           "function pointers were modified after driver loading."),
               fn_name);
}

using cuGetProcAddressT = CUresult (*)(const char*, void**, int, std::uint64_t);

[[nodiscard]] void* load_cu_driver_function_impl(cuGetProcAddressT cu_get_proc_address,
                                                 const char driver_function[],
                                                 int cuda_version,
                                                 bool can_fail)
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    const auto df_sv = std::string_view{driver_function};

    if (std::cmatch match{};
        std::regex_search(df_sv.cbegin(), df_sv.cend(), match, std::regex{"_v\\d+$"})) {
      throw legate::detail::TracedException<std::invalid_argument>{fmt::format(
        "CUDA driver symbol '{}' ill-formed, must not end in version specifier (found {})",
        df_sv,
        fmt::streamed(match[0]))};
    }
  }

  // clang-tidy says:
  //
  // src/cpp/legate/cuda/detail/cuda_driver_api.cc:140:3: error: pointee of variable 'ret' of
  // type 'void *' can be declared 'const' [misc-const-correctness,-warnings-as-errors]
  // 140 |   void* ret = nullptr;
  //     |   ^
  //     |        const
  //
  // But cu_get_proc_address() explicitly takes a void **, and we return a void *, so this
  // seems incorrect.
  void* ret = nullptr;  // NOLINT(misc-const-correctness)
  const auto error =
    cu_get_proc_address(driver_function, &ret, cuda_version, CU_GET_PROC_ADDRESS_DEFAULT);

  if (!can_fail && (error || !ret)) {
    throw legate::detail::TracedException<std::runtime_error>{
      fmt::format("Failed to load the symbol {} from the CUDA driver shared library: {}",
                  driver_function,
                  error)};
  }
  return ret;
}

template <typename F>
void load_cu_driver_function(cuGetProcAddressT cu_get_proc_address,
                             const char driver_function[],
                             F** dest_function,
                             int cuda_version,
                             bool can_fail)
{
  *dest_function = reinterpret_cast<F*>(
    load_cu_driver_function_impl(cu_get_proc_address, driver_function, cuda_version, can_fail));
}

}  // namespace

void CUDADriverAPI::read_symbols_()
{
  // Don't check that cuGetProcAddress is ABI compatible. We are specifically using the _v1
  // variant, which omits an argument we don't care about. If we were to use the
  // CHECK_ABI_COMPATIBLE() macro, then "cuGetProcAddress" would expand to
  // "cuGetProcAddress_v2" (since cuGetProcAddress is actually a macro!), and as explained
  // above, we aren't compatible with that one.
  //
  // Load this first via dlsym()...
  lib_.load_symbol_into("cuGetProcAddress", &get_proc_address_);
  // ...then get the rest through it. We do this to make sure we are always loading the latest
  // version of the API with which we are compatible.

// helper macro hacks for conditional compilation
#define LEGATE_IF(cond) LEGATE_IF_##cond
#define LEGATE_IF_0(t, f) f
#define LEGATE_IF_1(t, f) t

#define LEGATE_EAT(...)
#define LEGATE_EXPAND(...) __VA_ARGS__
#define LEGATE_WHEN(cond) LEGATE_IF(cond)(LEGATE_EXPAND, LEGATE_EAT)

#if LEGATE_CUDA_VERSION >= 13000
#define LEGATE_CU12 0
#define LEGATE_CU13 1
#elif (LEGATE_CUDA_VERSION >= 12000)
#define LEGATE_CU12 1
#define LEGATE_CU13 0
#endif

/**
 * @brief Load a CUDA driver function via cuGetProcAddress using current CUDA version.
 *
 * @param get_proc_address_ Function pointer to cuGetProcAddress from the CUDA driver
 * @param CU_FUNCTION The base name of the CUDA driver API function to load
 * @param OUR_FUNCTION Address of the function pointer to write the function pointer to
 */
#define LOAD_CU_DRIVER_FUNCTION(get_proc_address_, CU_FUNCTION, OUR_FUNCTION)                    \
  do {                                                                                           \
    /* This decltype trick is needed because referencing member objects (even if */              \
    /* just for the  purposes of deducing their type) is not considered a core */                \
    /* constant expression, since the member objects themselves would depend on the */           \
    /* address of the this pointer. So we get around this by creating a new function */          \
    /* pointer object of the same type. */                                                       \
    CHECK_ABI_COMPATIBLE(CU_FUNCTION, std::decay_t<decltype(*(OUR_FUNCTION))>{});                \
    /* Don't use LEGATE_STRINGIZE(), we don't want the CU_FUNCTION macro to expand */            \
    load_cu_driver_function(                                                                     \
      get_proc_address_, #CU_FUNCTION, OUR_FUNCTION, LEGATE_CUDA_VERSION, /* can_fail */ false); \
  } while (0)

/**
 * @brief Load a versioned CUDA driver function via cuGetProcAddress using a specific CUDA version.
 *
 * With CUDA 13.x, cuGetProcAddress broke its source compatibility requirements. This would lead
 * it to return e.g. cuCtxGetDevice_v2 instead of cuCtxGetDevice. This macro separates out the
 * versioned function for type-checking from the one provided to cuGetProcAddress and allows for
 * loading specific to a CUDA version.
 *
 * @note The CUDA_VERSION_ARG that is passed must be the largest of (a) the CUDA version
 * at which the desired symbol (CU_FUNCTION ## VER_SUFFIX) was introduced or (b) the
 * last major version the code was compiled with (e.g. (CUDA_VERSION / 1000) * 1000).
 *
 * This is because cuGetProcAddress will always return the symbol for the version passed to it.
 * However, the symbol located in cuda.h will be the same symbol as the last major version
 * the code was compiled with (e.g. symbols in CUDA 12.X will match symbols in CUDA 12.0)
 * Therefore, a signature mismatch can occur if we naively passed the CUDA_VERSION.
 *
 * @param get_proc_address_ Function pointer to cuGetProcAddress from the CUDA driver
 * @param CU_FUNCTION The base name of the CUDA driver API function to load
 * @param VER_SUFFIX The version suffix of the CUDA driver API function to load
 * @param OUR_FUNCTION Address of the function pointer to write the function pointer to
 * @param CUDA_VERSION_ARG The CUDA version to pass to cuGetProcAddress
 * @param ENABLE_ABI_CHECK Whether to check ABI compatibility
 */
#define LOAD_CU_DRIVER_FUNCTION_VER(                                                         \
  get_proc_address_, CU_FUNC, VER_SUFFIX, OUR_FUNCTION, CUDA_VERSION_ARG, ENABLE_ABI_CHECK)  \
  do {                                                                                       \
    LEGATE_WHEN(ENABLE_ABI_CHECK)(                                                           \
      CHECK_ABI_COMPATIBLE(CU_FUNC##VER_SUFFIX, std::decay_t<decltype(*(OUR_FUNCTION))>{})); \
    load_cu_driver_function(                                                                 \
      get_proc_address_, #CU_FUNC, OUR_FUNCTION, CUDA_VERSION_ARG, /* can_fail */ true);     \
  } while (0)

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuInit, &init_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuGetErrorString, &get_error_string_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuGetErrorName, &get_error_name_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuMemAllocAsync, &mem_alloc_async_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuMemFreeAsync, &mem_free_async_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuMemcpyAsync, &mem_cpy_async_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamCreate, &stream_create_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamDestroy, &stream_destroy_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamWaitEvent, &stream_wait_event_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamSynchronize, &stream_synchronize_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventCreate, &event_create_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventRecord, &event_record_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventSynchronize, &event_synchronize_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventElapsedTime, &event_elapsed_time_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventDestroy, &event_destroy_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuDevicePrimaryCtxRetain, &device_primary_ctx_retain_);
  LOAD_CU_DRIVER_FUNCTION(
    get_proc_address_, cuDevicePrimaryCtxRelease, &device_primary_ctx_release_);

  LOAD_CU_DRIVER_FUNCTION_VER(
    get_proc_address_, cuCtxGetDevice, /* no ver */, &ctx_get_device_cu_12_, 12000, LEGATE_CU12);
  LOAD_CU_DRIVER_FUNCTION_VER(
    get_proc_address_, cuCtxGetDevice, _v2, &ctx_get_device_cu_13_, 13000, LEGATE_CU13);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxGetCurrent, &ctx_get_current_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxPushCurrent, &ctx_push_current_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxPopCurrent, &ctx_pop_current_);
  LOAD_CU_DRIVER_FUNCTION_VER(
    get_proc_address_, cuCtxSynchronize, /* no ver */, &ctx_synchronize_cu_12_, 12000, LEGATE_CU12);
  LOAD_CU_DRIVER_FUNCTION_VER(
    get_proc_address_, cuCtxSynchronize, _v2, &ctx_synchronize_cu_13_, 13000, LEGATE_CU13);
  if (!ctx_get_device_cu_12_ && !ctx_get_device_cu_13_) {
    throw_null_driver_request("cuCtxGetDevice");
  }
  if (!ctx_synchronize_cu_12_ && !ctx_synchronize_cu_13_) {
    throw_null_driver_request("cuCtxSynchronize");
  }

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuKernelGetFunction, &kernel_get_function_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLaunchKernel, &launch_kernel_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryLoadData, &library_load_data_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryGetKernel, &library_get_kernel_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryUnload, &library_unload_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuMemGetInfo, &mem_get_info_);
}

void CUDADriverAPI::check_initialized_() const
{
  if (!is_loaded()) {
    throw legate::detail::TracedException<std::logic_error>{
      fmt::format("Cannot call CUDA driver API, failed to load {}", handle_path())};
  }
}

// ==========================================================================================

CUDADriverAPI::CUDADriverAPI(std::string handle_path)
  : lib_{std::move(handle_path), /* must_load */ false}
{
  if (is_loaded()) {
    read_symbols_();
  }
}

#define LEGATE_CHECK_CUDRIVER(...)                                                          \
  do {                                                                                      \
    const ::legate::CUresult __legate_cu_result__ = __VA_ARGS__;                            \
    if (LEGATE_UNLIKELY(__legate_cu_result__)) {                                            \
      ::legate::cuda::detail::throw_cuda_driver_error(                                      \
        __legate_cu_result__, LEGATE_STRINGIZE(__VA_ARGS__), __FILE__, __func__, __LINE__); \
    }                                                                                       \
  } while (0)

void CUDADriverAPI::init() const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(init_(0));
}

const char* CUDADriverAPI::get_error_string(CUresult error) const
{
  const char* str;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(get_error_string_(error, &str));
  return str;
}

const char* CUDADriverAPI::get_error_name(CUresult error) const
{
  const char* str;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(get_error_name_(error, &str));
  return str;
}

void* CUDADriverAPI::mem_alloc_async(std::size_t num_bytes, CUstream stream) const
{
  CUdeviceptr ret{};

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(mem_alloc_async_(&ret, num_bytes, stream));
  return reinterpret_cast<void*>(ret);  // NOLINT(performance-no-int-to-ptr)
}

void CUDADriverAPI::mem_free_async(void** ptr, CUstream stream) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(mem_free_async_(reinterpret_cast<CUdeviceptr>(*ptr), stream));
  *ptr = nullptr;
}

void CUDADriverAPI::mem_cpy_async(CUdeviceptr dst,
                                  CUdeviceptr src,
                                  std::size_t num_bytes,
                                  CUstream stream) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(mem_cpy_async_(dst, src, num_bytes, stream));
}

void CUDADriverAPI::mem_cpy_async(void* dst,
                                  const void* src,
                                  std::size_t num_bytes,
                                  CUstream stream) const
{
  mem_cpy_async(
    reinterpret_cast<CUdeviceptr>(dst), reinterpret_cast<CUdeviceptr>(src), num_bytes, stream);
}

CUstream CUDADriverAPI::stream_create(unsigned int flags) const
{
  CUstream stream;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(stream_create_(&stream, flags));
  return stream;
}

void CUDADriverAPI::stream_destroy(CUstream* stream) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(stream_destroy_(*stream));
  *stream = nullptr;
}

void CUDADriverAPI::stream_synchronize(CUstream stream) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(stream_synchronize_(stream));
}

void CUDADriverAPI::stream_wait_event(CUstream stream, CUevent event, unsigned int flags) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(stream_wait_event_(stream, event, flags));
}

CUevent CUDADriverAPI::event_create(unsigned int flags) const
{
  CUevent event;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(event_create_(&event, flags));
  return event;
}

void CUDADriverAPI::event_record(CUevent event, CUstream stream) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(event_record_(event, stream));
}

void CUDADriverAPI::event_synchronize(CUevent event) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(event_synchronize_(event));
}

float CUDADriverAPI::event_elapsed_time(CUevent start, CUevent end) const
{
  float ms;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(event_elapsed_time_(&ms, start, end));
  return ms;
}

void CUDADriverAPI::event_destroy(CUevent* event) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(event_destroy_(*event));
  *event = nullptr;
}

CUcontext CUDADriverAPI::device_primary_ctx_retain(CUdevice dev) const
{
  CUcontext ctx;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(device_primary_ctx_retain_(&ctx, dev));
  return ctx;
}

void CUDADriverAPI::device_primary_ctx_release(CUdevice dev) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(device_primary_ctx_release_(dev));
}

CUcontext CUDADriverAPI::ctx_get_current() const
{
  CUcontext ctx;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(ctx_get_current_(&ctx));
  return ctx;
}

CUdevice CUDADriverAPI::ctx_get_device(CUcontext ctx) const
{
  CUdevice device;

  check_initialized_();
  if (ctx_get_device_cu_13_) {  // loaded CUDA 13.x at runtime
    LEGATE_CHECK_CUDRIVER(ctx_get_device_cu_13_(&device, ctx));
  } else if (ctx_get_device_cu_12_) {  // loaded CUDA 12.x at runtime
    const AutoCUDAContext auto_ctx{ctx};

    LEGATE_CHECK_CUDRIVER(ctx_get_device_cu_12_(&device));
  } else {
    abort_null_driver_fn("cuCtxGetDevice");
  }

  return device;
}

void CUDADriverAPI::ctx_push_current(CUcontext ctx) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(ctx_push_current_(ctx));
}

CUcontext CUDADriverAPI::ctx_pop_current() const
{
  CUcontext ctx;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(ctx_pop_current_(&ctx));
  return ctx;
}

void CUDADriverAPI::ctx_synchronize(CUcontext ctx) const
{
  check_initialized_();
  if (ctx_synchronize_cu_13_) {  // loaded CUDA 13.x at runtime
    LEGATE_CHECK_CUDRIVER(ctx_synchronize_cu_13_(ctx));
  } else if (ctx_synchronize_cu_12_) {  // loaded CUDA 12.x at runtime
    const AutoCUDAContext auto_ctx{ctx};

    LEGATE_CHECK_CUDRIVER(ctx_synchronize_cu_12_());
  } else {
    abort_null_driver_fn("cuCtxSynchronize");
  }
}

CUfunction CUDADriverAPI::kernel_get_function(CUkernel kernel) const
{
  CUfunction func;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(kernel_get_function_(&func, kernel));
  return func;
}

void CUDADriverAPI::launch_kernel_direct(CUfunction f,
                                         Dim3 grid_dim,
                                         Dim3 block_dim,
                                         std::size_t shared_mem_bytes,
                                         CUstream stream,
                                         void** kernel_params,
                                         void** extra) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(launch_kernel_(f,
                                       static_cast<unsigned int>(grid_dim.x),
                                       static_cast<unsigned int>(grid_dim.y),
                                       static_cast<unsigned int>(grid_dim.z),
                                       static_cast<unsigned int>(block_dim.x),
                                       static_cast<unsigned int>(block_dim.y),
                                       static_cast<unsigned int>(block_dim.z),
                                       static_cast<unsigned int>(shared_mem_bytes),
                                       stream,
                                       kernel_params,
                                       extra));
}

CUlibrary CUDADriverAPI::library_load_data(const void* code,
                                           CUjit_option* jit_options,
                                           void** jit_options_values,
                                           std::size_t num_jit_options,
                                           CUlibraryOption* library_options,
                                           void** library_option_values,
                                           std::size_t num_library_options) const
{
  CUlibrary library;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(library_load_data_(&library,
                                           code,
                                           jit_options,
                                           jit_options_values,
                                           static_cast<unsigned int>(num_jit_options),
                                           library_options,
                                           library_option_values,
                                           static_cast<unsigned int>(num_library_options)));
  return library;
}

CUkernel CUDADriverAPI::library_get_kernel(CUlibrary library, const char* name) const
{
  CUkernel kernel;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(library_get_kernel_(&kernel, library, name));
  return kernel;
}

void CUDADriverAPI::library_unload(CUlibrary* library) const
{
  check_initialized_();
  LEGATE_CHECK_CUDRIVER(library_unload_(*library));
  *library = nullptr;
}

std::pair<std::size_t, std::size_t> CUDADriverAPI::mem_get_info() const
{
  std::size_t free, total;

  check_initialized_();
  LEGATE_CHECK_CUDRIVER(mem_get_info_(&free, &total));
  return {free, total};
}

// ==========================================================================================

legate::detail::ZStringView CUDADriverAPI::handle_path() const noexcept
{
  return lib_.handle_path();
}

bool CUDADriverAPI::is_loaded() const noexcept { return lib_.is_loaded(); }

// ==========================================================================================

namespace {

[[noreturn]] void abort_expected_context(CUcontext active_ctx, CUcontext expected_ctx)
{
  LEGATE_ABORT("Current active CUDA context ",
               active_ctx,
               " does not match expected context ",
               expected_ctx,
               " (the primary context retained by this object). This indicates another CUDA "
               "context was pushed onto the stack during the lifetime of this object, without "
               "being popped from the stack.");
}

}  // namespace

AutoPrimaryContext::AutoPrimaryContext()
  : AutoPrimaryContext{legate::detail::Runtime::get_runtime().get_current_cuda_device()}
{
}

AutoPrimaryContext::AutoPrimaryContext(CUdevice device) : device_{device}
{
  auto&& api = get_cuda_driver_api();

  ctx_ = api->device_primary_ctx_retain(device_);
  api->ctx_push_current(ctx_);
}

AutoPrimaryContext::~AutoPrimaryContext()
{
  try {
    auto&& api = get_cuda_driver_api();

    if (const auto cur_ctx = api->ctx_pop_current(); cur_ctx != ctx_) {
      abort_expected_context(cur_ctx, ctx_);
    }
    api->device_primary_ctx_release(device_);
  } catch (const std::exception& e) {
    LEGATE_ABORT(e.what());
  }
}

// ==========================================================================================

AutoCUDAContext::AutoCUDAContext(CUcontext ctx) : ctx_{ctx}
{
  get_cuda_driver_api()->ctx_push_current(ctx_);
}

AutoCUDAContext::~AutoCUDAContext()
{
  try {
    auto&& api = get_cuda_driver_api();

    if (const auto cur_ctx = api->ctx_pop_current(); cur_ctx != ctx_) {
      abort_expected_context(cur_ctx, ctx_);
    }
  } catch (const std::exception& e) {
    LEGATE_ABORT(e.what());
  }
}

// ==========================================================================================

CUDADriverError::CUDADriverError(const std::string& what, CUresult result)
  : runtime_error{what}, result_{result}
{
}

CUresult CUDADriverError::error_code() const noexcept { return result_; }

// ==========================================================================================

namespace {

std::optional<InternalSharedPtr<CUDADriverAPI>> CUDA_DRIVER_API{};

}  // namespace

// These functions could technically race, if one thread sets the active driver API while
// another is retrieving it. But we (currently) only set the driver path at legate startup
// where a single thread reigns supreme, so there doesn't seem much point in guarding access
// with a mutex here.
void set_active_cuda_driver_api(std::string handle_path)
{
  if (CUDA_DRIVER_API.has_value() && (*CUDA_DRIVER_API)->handle_path() == handle_path) {
    return;
  }
  CUDA_DRIVER_API = make_internal_shared<CUDADriverAPI>(std::move(handle_path));
}

const InternalSharedPtr<CUDADriverAPI>& get_cuda_driver_api()
{
  if (!CUDA_DRIVER_API.has_value()) {
    throw legate::detail::TracedException<std::runtime_error>{
      "Must initialize Legate via legate::start() before making CUDA driver calls"};
  }
  return *CUDA_DRIVER_API;
}

// ==========================================================================================

void throw_cuda_driver_error(CUresult result,
                             std::string_view expression,
                             std::string_view file,
                             std::string_view func,
                             int line)
{
  const char* error_str = [&] {
    // Do not care about the error, in fact, cannot handle it.
    try {
      return get_cuda_driver_api()->get_error_string(result);
    } catch (...) {
      return "unknown error occurred";
    }
  }();

  const char* error_name = [&] {
    try {
      return get_cuda_driver_api()->get_error_name(result);
    } catch (...) {
      return "unknown error";
    }
  }();

  throw legate::detail::TracedException<CUDADriverError>{
    fmt::format("CUDA driver expression '{}' failed at {}:{} (in {}()) with error code {} ({}): {}",
                expression,
                file,
                line,
                func,
                result,
                error_name,
                error_str),
    result};
}

}  // namespace legate::cuda::detail
