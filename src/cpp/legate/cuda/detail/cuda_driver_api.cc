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

#include "legate/runtime/detail/runtime.h"
#include "legate/utilities/assert.h"
#include "legate/utilities/detail/env.h"
#include "legate/utilities/macros.h"

#include <dlfcn.h>
#include <fmt/format.h>
#include <stdexcept>

namespace legate::cuda::detail {

void CUDADriverAPI::read_symbols_()
{
#define LEGATE_CU_LOAD_FN(member_function, driver_function)                     \
  do {                                                                          \
    static_cast<void>(::dlerror());                                             \
    this->member_function = reinterpret_cast<decltype(this->member_function)>(  \
      ::dlsym(handle_, LEGATE_STRINGIZE_(driver_function)));                    \
    if (const char* error = dlerror()) {                                        \
      throw std::runtime_error{                                                 \
        fmt::format("Failed to locate the symbol {} in the shared library: {}", \
                    LEGATE_STRINGIZE_(driver_function),                         \
                    error)};                                                    \
    }                                                                           \
  } while (0)

  LEGATE_CU_LOAD_FN(init_, cuInit);

  LEGATE_CU_LOAD_FN(get_error_string_, cuGetErrorString);
  LEGATE_CU_LOAD_FN(get_error_name_, cuGetErrorName);

  LEGATE_CU_LOAD_FN(pointer_get_attributes_, cuPointerGetAttributes);

  LEGATE_CU_LOAD_FN(mem_cpy_async_, cuMemcpyAsync);
  LEGATE_CU_LOAD_FN(mem_cpy_, cuMemcpy);

  LEGATE_CU_LOAD_FN(stream_create_, cuStreamCreate);
  LEGATE_CU_LOAD_FN(stream_destroy_, cuStreamDestroy);
  LEGATE_CU_LOAD_FN(stream_synchronize_, cuStreamSynchronize);

  LEGATE_CU_LOAD_FN(event_create_, cuEventCreate);
  LEGATE_CU_LOAD_FN(event_record_, cuEventRecord);
  LEGATE_CU_LOAD_FN(event_synchronize_, cuEventSynchronize);
  LEGATE_CU_LOAD_FN(event_elapsed_time_, cuEventElapsedTime);
  LEGATE_CU_LOAD_FN(event_destroy_, cuEventDestroy);

  LEGATE_CU_LOAD_FN(ctx_get_device_, cuCtxGetDevice);
  LEGATE_CU_LOAD_FN(ctx_synchronize_, cuCtxSynchronize);

  LEGATE_CU_LOAD_FN(launch_kernel_, cuLaunchKernel);

  LEGATE_CU_LOAD_FN(library_load_data_, cuLibraryLoadData);
  LEGATE_CU_LOAD_FN(library_get_kernel_, cuLibraryGetKernel);
  LEGATE_CU_LOAD_FN(library_unload_, cuLibraryUnload);
#undef LEGATE_CU_LOAD_FN
}

void CUDADriverAPI::check_initialized_() const
{
  if (!is_loaded()) {
    throw std::logic_error{"Cannot call CUDA driver API, failed to load libcuda.so"};
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
  if (::Dl_info info{}; ::dladdr(reinterpret_cast<const void*>(init_), &info)) {
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

CUresult CUDADriverAPI::pointer_get_attributes(void* data, int attribute, CUdeviceptr ptr) const
{
  check_initialized_();
  return pointer_get_attributes_(data, attribute, ptr);
}

CUresult CUDADriverAPI::mem_cpy_async(CUdeviceptr dst,
                                      CUdeviceptr src,
                                      std::size_t num_bytes,
                                      CUstream stream) const
{
  check_initialized_();
  return mem_cpy_async_(dst, src, num_bytes, stream);
}

CUresult CUDADriverAPI::mem_cpy(CUdeviceptr dst, CUdeviceptr src, std::size_t num_bytes) const
{
  check_initialized_();
  return mem_cpy_(dst, src, num_bytes);
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

CUresult CUDADriverAPI::ctx_get_device(CUdevice* device) const
{
  check_initialized_();
  return ctx_get_device_(device);
}

CUresult CUDADriverAPI::ctx_synchronize() const
{
  check_initialized_();
  return ctx_synchronize_();
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
  throw CUDADriverError{
    fmt::format("Expression '{}' failed at {}:{} (in {}()) with error code {} ({})",
                expression,
                file,
                line,
                func,
                static_cast<int>(result),
                error_str),
    result};
}

}  // namespace legate::cuda::detail
