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

#include "legate/utilities/assert.h"
#include "legate/utilities/detail/env.h"
#include "legate/utilities/macros.h"

#include <dlfcn.h>
#include <fmt/format.h>
#include <stdexcept>

namespace legate::cuda::detail {

void CUDADriverAPI::read_symbols_()
{
#define LEGATE_CU_LOAD_FN(name, function)                                       \
  do {                                                                          \
    static_cast<void>(::dlerror());                                             \
    this->LEGATE_CONCAT_(function, _) =                                         \
      reinterpret_cast<decltype(this->LEGATE_CONCAT_(function, _))>(            \
        ::dlsym(handle_, "cu" LEGATE_STRINGIZE_(name)));                        \
    if (const char* error = dlerror(); error) {                                 \
      throw std::runtime_error{                                                 \
        fmt::format("Failed to locate the symbol {} in the shared library: {}", \
                    LEGATE_STRINGIZE_(name),                                    \
                    error)};                                                    \
    }                                                                           \
  } while (0)

  LEGATE_CU_LOAD_FN(Init, init);
  LEGATE_CU_LOAD_FN(StreamCreate, stream_create);
  LEGATE_CU_LOAD_FN(GetErrorString, get_error_string);
  LEGATE_CU_LOAD_FN(GetErrorName, get_error_name);
  LEGATE_CU_LOAD_FN(PointerGetAttributes, pointer_get_attributes);
  LEGATE_CU_LOAD_FN(MemcpyAsync, mem_cpy_async);
  LEGATE_CU_LOAD_FN(Memcpy, mem_cpy);
  LEGATE_CU_LOAD_FN(StreamDestroy, stream_destroy);
  LEGATE_CU_LOAD_FN(StreamSynchronize, stream_synchronize);
  LEGATE_CU_LOAD_FN(CtxSynchronize, ctx_synchronize);

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

CUresult CUDADriverAPI::stream_create(CUstream* stream, unsigned int flags) const
{
  check_initialized_();
  return stream_create_(stream, flags);
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

CUresult CUDADriverAPI::ctx_synchronize() const
{
  check_initialized_();
  return ctx_synchronize_();
}

std::string_view CUDADriverAPI::handle_path() const noexcept { return handle_path_; }

bool CUDADriverAPI::is_loaded() const noexcept { return handle_ != nullptr; }

}  // namespace legate::cuda::detail
