/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/cuda/detail/cuda_driver_api.h"

#include <legate_defines.h>

#include "legate/runtime/detail/runtime.h"
#include "legate/utilities/assert.h"
#include "legate/utilities/detail/env.h"
#include <legate/utilities/macros.h>

#include <dlfcn.h>
#include <fmt/format.h>
#include <fmt/ostream.h>
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

namespace {

namespace abi_detail {

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

}  // namespace abi_detail

}  // namespace

namespace legate::cuda::detail {

namespace {

template <typename F>
void load_dlsym_function(void* handle, const char name[], F** dest)
{
  static_cast<void>(::dlerror());
  *dest = reinterpret_cast<F*>(::dlsym(handle, name));
  if (const char* error = ::dlerror()) {
    throw std::runtime_error{
      fmt::format("Failed to locate the symbol {} in the shared library: {}", name, error)};
  }
}

using cuGetProcAddressT = CUresult (*)(const char*, void**, int, std::uint64_t);

[[nodiscard]] void* load_cu_driver_function_impl(cuGetProcAddressT cu_get_proc_address,
                                                 const char driver_function[])
{
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    const auto df_sv = std::string_view{driver_function};

    if (std::cmatch match{};
        std::regex_search(df_sv.cbegin(), df_sv.cend(), match, std::regex{"_v\\d+$"})) {
      throw std::invalid_argument{fmt::format(
        "CUDA driver symbol '{}' ill-formed, must not end in version specifier (found {})",
        df_sv,
        fmt::streamed(match[0]))};
    }
  }

  void* ret = nullptr;
  const auto error =
    cu_get_proc_address(driver_function, &ret, LEGATE_CUDA_VERSION, CU_GET_PROC_ADDRESS_DEFAULT);

  if (error || !ret) {
    throw std::runtime_error{
      fmt::format("Failed to load the symbol {} from the CUDA driver shared library: {}",
                  driver_function,
                  error)};
  }
  return ret;
}

template <typename F>
void load_cu_driver_function(cuGetProcAddressT cu_get_proc_address,
                             const char driver_function[],
                             F** dest_function)
{
  *dest_function =
    reinterpret_cast<F*>(load_cu_driver_function_impl(cu_get_proc_address, driver_function));
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
  load_dlsym_function(handle_, "cuGetProcAddress", &get_proc_address_);
  // ...then get the rest through it. We do this to make sure we are always loading the latest
  // version of the API with which we are compatible.

#define LOAD_CU_DRIVER_FUNCTION(get_proc_address_, CU_FUNCTION, OUR_FUNCTION)           \
  do {                                                                                  \
    /* This decltype trick is needed because referencing member objects (even if */     \
    /* just for the  purposes of deducing their type) is not considered a core */       \
    /* constant expression, since the member objects themselves would depend on the */  \
    /* address of the this pointer. So we get around this by creating a new function */ \
    /* pointer object of the same type. */                                              \
    CHECK_ABI_COMPATIBLE(CU_FUNCTION, std::decay_t<decltype(*(OUR_FUNCTION))>{});       \
    /* Don't use LEGATE_STRINGIZE(), we don't want the CU_FUNCTION macro to expand */   \
    load_cu_driver_function(get_proc_address_, #CU_FUNCTION, OUR_FUNCTION);             \
  } while (0)

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuInit, &init_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuGetErrorString, &get_error_string_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuGetErrorName, &get_error_name_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuMemcpyAsync, &mem_cpy_async_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamCreate, &stream_create_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamDestroy, &stream_destroy_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuStreamSynchronize, &stream_synchronize_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventCreate, &event_create_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventRecord, &event_record_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventSynchronize, &event_synchronize_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventElapsedTime, &event_elapsed_time_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuEventDestroy, &event_destroy_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuDevicePrimaryCtxRetain, &device_primary_ctx_retain_);
  LOAD_CU_DRIVER_FUNCTION(
    get_proc_address_, cuDevicePrimaryCtxRelease, &device_primary_ctx_release_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxGetDevice, &ctx_get_device_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxPushCurrent, &ctx_push_current_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxPopCurrent, &ctx_pop_current_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuCtxSynchronize, &ctx_synchronize_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuKernelGetFunction, &kernel_get_function_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLaunchKernel, &launch_kernel_);

  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryLoadData, &library_load_data_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryGetKernel, &library_get_kernel_);
  LOAD_CU_DRIVER_FUNCTION(get_proc_address_, cuLibraryUnload, &library_unload_);
}

void CUDADriverAPI::check_initialized_() const
{
  if (!is_loaded()) {
    throw std::logic_error{
      fmt::format("Cannot call CUDA driver API, failed to load {}", handle_path())};
  }
}

// ==========================================================================================

CUDADriverAPI::CUDADriverAPI()
  : handle_path_{legate::detail::LEGATE_CUDA_DRIVER.get(/* default */ "libcuda.so.1")},
    handle_{::dlopen(handle_path_.c_str(), RTLD_LAZY | RTLD_LOCAL)}
{
  if (!is_loaded()) {
    return;
  }
  read_symbols_();
  // Can only do this with get_proc_address_, because that's the only symbol we acquire through
  // dlsym() directly.
  if (::Dl_info info{}; ::dladdr(reinterpret_cast<const void*>(get_proc_address_), &info)) {
    LEGATE_CHECK(info.dli_fname);
    handle_path_ = info.dli_fname;
  }
}

CUresult CUDADriverAPI::init(unsigned int flags) const
{
  check_initialized_();
  return init_(flags);
}

CUresult CUDADriverAPI::get_error_string(CUresult error, const char** str) const
{
  check_initialized_();
  return get_error_string_(error, str);
}

CUresult CUDADriverAPI::get_error_name(CUresult error, const char** str) const
{
  check_initialized_();
  return get_error_name_(error, str);
}

CUresult CUDADriverAPI::mem_cpy_async(CUdeviceptr dst,
                                      CUdeviceptr src,
                                      std::size_t num_bytes,
                                      CUstream stream) const
{
  check_initialized_();
  return mem_cpy_async_(dst, src, num_bytes, stream);
}

CUresult CUDADriverAPI::stream_create(CUstream* stream, unsigned int flags) const
{
  check_initialized_();
  return stream_create_(stream, flags);
}

CUresult CUDADriverAPI::stream_destroy(CUstream stream) const
{
  check_initialized_();
  return stream_destroy_(stream);
}

CUresult CUDADriverAPI::stream_synchronize(CUstream stream) const
{
  check_initialized_();
  return stream_synchronize_(stream);
}

CUresult CUDADriverAPI::event_create(CUevent* event, unsigned int flags) const
{
  check_initialized_();
  return event_create_(event, flags);
}

CUresult CUDADriverAPI::event_record(CUevent event, CUstream stream) const
{
  check_initialized_();
  return event_record_(event, stream);
}

CUresult CUDADriverAPI::event_synchronize(CUevent event) const
{
  check_initialized_();
  return event_synchronize_(event);
}

CUresult CUDADriverAPI::event_elapsed_time(float* ms, CUevent start, CUevent end) const
{
  check_initialized_();
  return event_elapsed_time_(ms, start, end);
}

CUresult CUDADriverAPI::event_destroy(CUevent event) const
{
  check_initialized_();
  return event_destroy_(event);
}

CUresult CUDADriverAPI::device_primary_ctx_retain(CUcontext* ctx, CUdevice dev) const
{
  check_initialized_();
  return device_primary_ctx_retain_(ctx, dev);
}

CUresult CUDADriverAPI::device_primary_ctx_release(CUdevice dev) const
{
  check_initialized_();
  return device_primary_ctx_release_(dev);
}

CUresult CUDADriverAPI::ctx_get_device(CUdevice* device) const
{
  check_initialized_();
  return ctx_get_device_(device);
}

CUresult CUDADriverAPI::ctx_push_current(CUcontext ctx) const
{
  check_initialized_();
  return ctx_push_current_(ctx);
}

CUresult CUDADriverAPI::ctx_pop_current(CUcontext* ctx) const
{
  check_initialized_();
  return ctx_pop_current_(ctx);
}

CUresult CUDADriverAPI::ctx_synchronize() const
{
  check_initialized_();
  return ctx_synchronize_();
}

CUresult CUDADriverAPI::kernel_get_function(CUfunction* func, CUkernel kernel) const
{
  check_initialized_();
  return kernel_get_function_(func, kernel);
}

CUresult CUDADriverAPI::launch_kernel(CUfunction f,
                                      Dim3 grid_dim,
                                      Dim3 block_dim,
                                      std::size_t shared_mem_bytes,
                                      CUstream stream,
                                      void** kernel_params,
                                      void** extra) const
{
  check_initialized_();
  return launch_kernel_(f,
                        static_cast<unsigned int>(grid_dim.x),
                        static_cast<unsigned int>(grid_dim.y),
                        static_cast<unsigned int>(grid_dim.z),
                        static_cast<unsigned int>(block_dim.x),
                        static_cast<unsigned int>(block_dim.y),
                        static_cast<unsigned int>(block_dim.z),
                        static_cast<unsigned int>(shared_mem_bytes),
                        stream,
                        kernel_params,
                        extra);
}

CUresult CUDADriverAPI::launch_kernel(CUkernel f,
                                      Dim3 grid_dim,
                                      Dim3 block_dim,
                                      std::size_t shared_mem_bytes,
                                      CUstream stream,
                                      void** kernel_params,
                                      void** extra) const
{
  return launch_kernel(reinterpret_cast<CUfunction>(f),
                       grid_dim,
                       block_dim,
                       shared_mem_bytes,
                       stream,
                       kernel_params,
                       extra);
}

CUresult CUDADriverAPI::library_load_data(CUlibrary* library,
                                          const void* code,
                                          CUjit_option* jit_options,
                                          void** jit_options_values,
                                          std::size_t num_jit_options,
                                          CUlibraryOption* library_options,
                                          void** library_option_values,
                                          std::size_t num_library_options) const
{
  check_initialized_();
  return library_load_data_(library,
                            code,
                            jit_options,
                            jit_options_values,
                            static_cast<unsigned int>(num_jit_options),
                            library_options,
                            library_option_values,
                            static_cast<unsigned int>(num_library_options));
}

CUresult CUDADriverAPI::library_get_kernel(CUkernel* kernel,
                                           CUlibrary library,
                                           const char* name) const
{
  check_initialized_();
  return library_get_kernel_(kernel, library, name);
}

CUresult CUDADriverAPI::library_unload(CUlibrary library) const
{
  check_initialized_();
  return library_unload_(library);
}

// ==========================================================================================

std::string_view CUDADriverAPI::handle_path() const noexcept { return handle_path_; }

bool CUDADriverAPI::is_loaded() const noexcept { return handle_ != nullptr; }

// ==========================================================================================

CUDADriverError::CUDADriverError(const std::string& what, CUresult result)
  : runtime_error{what}, result_{result}
{
}

CUresult CUDADriverError::error_code() const noexcept { return result_; }

// ==========================================================================================

void throw_cuda_driver_error(CUresult result,
                             std::string_view expression,
                             std::string_view file,
                             std::string_view func,
                             int line)
{
  const char* error_str{};

  // Do not care about the error, in fact, cannot handle it.
  try {
    static_cast<void>(
      legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->get_error_string(result,
                                                                                      &error_str));
  } catch (const std::exception& exn) {
    error_str = "unknown error occurred";
  }

  const char* error_name{};

  try {
    static_cast<void>(legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->get_error_name(
      result, &error_name));
  } catch (const std::exception& exn) {
    error_str = "unknown error";
  }

  throw CUDADriverError{
    fmt::format("Expression '{}' failed at {}:{} (in {}()) with error code {} ({}): {}",
                expression,
                file,
                line,
                func,
                static_cast<int>(result),
                error_name,
                error_str),
    result};
}

}  // namespace legate::cuda::detail
