/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/cuda/detail/cuda_driver_types.h>
#include <legate/utilities/detail/shared_library.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/macros.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace legate::cuda::detail {

class Dim3 {
 public:
  constexpr Dim3(std::size_t x_ = 1,  // NOLINT(google-explicit-constructor)
                 std::size_t y_ = 1,
                 std::size_t z_ = 1) noexcept
    : x{x_}, y{y_}, z{z_}
  {
  }

  std::size_t x{1};
  std::size_t y{1};
  std::size_t z{1};
};

class CUDADriverAPI {
 public:
  explicit CUDADriverAPI(std::string handle_path);

  void init() const;

  [[nodiscard]] const char* get_error_string(CUresult error) const;
  [[nodiscard]] const char* get_error_name(CUresult error) const;

  [[nodiscard]] void* mem_alloc_async(std::size_t num_bytes, CUstream stream) const;
  [[nodiscard]] void* mem_alloc_managed(std::size_t num_bytes,
                                        unsigned int flags = CU_MEM_ATTACH_GLOBAL) const;
  void mem_free_async(void** ptr, CUstream stream) const;
  void mem_cpy_async(CUdeviceptr dst,
                     CUdeviceptr src,
                     std::size_t num_bytes,
                     CUstream stream) const;
  void mem_cpy_async(void* dst, const void* src, std::size_t num_bytes, CUstream stream) const;

  [[nodiscard]] CUstream stream_create(unsigned int flags) const;
  void stream_destroy(CUstream* stream) const;
  void stream_wait_event(CUstream stream, CUevent event, unsigned int flags = 0) const;
  void stream_synchronize(CUstream stream) const;

  [[nodiscard]] CUevent event_create(unsigned int flags = 0) const;
  void event_record(CUevent event, CUstream stream) const;
  void event_synchronize(CUevent event) const;
  [[nodiscard]] CUresult event_query(CUevent event) const;
  [[nodiscard]] float event_elapsed_time(CUevent start, CUevent end) const;
  void event_destroy(CUevent* event) const;

  [[nodiscard]] CUcontext device_primary_ctx_retain(CUdevice dev) const;
  void device_primary_ctx_release(CUdevice dev) const;

  [[nodiscard]] CUcontext ctx_get_current() const;
  [[nodiscard]] CUdevice ctx_get_device(CUcontext ctx) const;
  void ctx_push_current(CUcontext ctx) const;
  [[nodiscard]] CUcontext ctx_pop_current() const;
  void ctx_synchronize(CUcontext ctx) const;
  void ctx_record_event(CUcontext ctx, CUevent event) const;

  [[nodiscard]] CUfunction kernel_get_function(CUkernel kernel) const;

  void launch_kernel_direct(CUfunction f,
                            Dim3 grid_dim,
                            Dim3 block_dim,
                            std::size_t shared_mem_bytes,
                            CUstream stream,
                            void** kernel_params,
                            void** extra) const;
  template <typename... T>
  void launch_kernel(CUkernel f,
                     Dim3 grid_dim,
                     Dim3 block_dim,
                     std::size_t shared_mem_bytes,
                     CUstream stream,
                     T&&... args) const;
  template <typename... T>
  void launch_kernel(CUfunction f,
                     Dim3 grid_dim,
                     Dim3 block_dim,
                     std::size_t shared_mem_bytes,
                     CUstream stream,
                     T&&... args) const;

  [[nodiscard]] CUlibrary library_load_data(const void* code,
                                            CUjit_option* jit_options,
                                            void** jit_options_values,
                                            std::size_t num_jit_options,
                                            CUlibraryOption* library_options,
                                            void** library_option_values,
                                            std::size_t num_library_options) const;
  [[nodiscard]] CUkernel library_get_kernel(CUlibrary library, const char* name) const;
  void library_unload(CUlibrary* library) const;

  [[nodiscard]] std::pair<std::size_t, std::size_t> mem_get_info() const;

  [[nodiscard]] legate::detail::ZStringView handle_path() const noexcept;
  [[nodiscard]] bool is_loaded() const noexcept;

 private:
  legate::detail::SharedLibrary lib_;

  void read_symbols_();
  void check_initialized_() const;

  CUresult (*get_proc_address_)(const char* symbol,
                                void** pfn,
                                int cuda_version,
                                std::uint64_t flags) = nullptr;

  CUresult (*init_)(unsigned int flags) = nullptr;

  CUresult (*get_error_string_)(CUresult error, const char** str) = nullptr;
  CUresult (*get_error_name_)(CUresult error, const char** str)   = nullptr;

  CUresult (*mem_alloc_async_)(CUdeviceptr* dptr, std::size_t num_bytes, CUstream stream) = nullptr;
  CUresult (*mem_alloc_managed_)(CUdeviceptr* dptr,
                                 std::size_t num_bytes,
                                 unsigned int flags)                                      = nullptr;
  CUresult (*mem_free_async_)(CUdeviceptr ptr, CUstream stream)                           = nullptr;
  CUresult (*mem_cpy_async_)(CUdeviceptr dst,
                             CUdeviceptr src,
                             std::size_t num_bytes,
                             CUstream stream)                                             = nullptr;

  CUresult (*stream_create_)(CUstream* stream, unsigned int flags)                   = nullptr;
  CUresult (*stream_destroy_)(CUstream stream)                                       = nullptr;
  CUresult (*stream_wait_event_)(CUstream stream, CUevent event, unsigned int flags) = nullptr;
  CUresult (*stream_synchronize_)(CUstream stream)                                   = nullptr;

  CUresult (*event_create_)(CUevent* event, unsigned int flags)          = nullptr;
  CUresult (*event_record_)(CUevent event, CUstream stream)              = nullptr;
  CUresult (*event_synchronize_)(CUevent event)                          = nullptr;
  CUresult (*event_query_)(CUevent event)                                = nullptr;
  CUresult (*event_elapsed_time_)(float* ms, CUevent start, CUevent end) = nullptr;
  CUresult (*event_destroy_)(CUevent event)                              = nullptr;

  CUresult (*device_primary_ctx_retain_)(CUcontext* ctx, CUdevice dev) = nullptr;
  CUresult (*device_primary_ctx_release_)(CUdevice dev)                = nullptr;

  CUresult (*ctx_get_current_)(CUcontext* ctx)                       = nullptr;
  CUresult (*ctx_get_device_cu_12_)(CUdevice* device)                = nullptr;
  CUresult (*ctx_get_device_cu_13_)(CUdevice* device, CUcontext ctx) = nullptr;
  CUresult (*ctx_push_current_)(CUcontext ctx)                       = nullptr;
  CUresult (*ctx_pop_current_)(CUcontext* ctx)                       = nullptr;
  CUresult (*ctx_synchronize_cu_12_)()                               = nullptr;
  CUresult (*ctx_synchronize_cu_13_)(CUcontext ctx)                  = nullptr;
  CUresult (*ctx_record_event_)(CUcontext ctx, CUevent event)        = nullptr;

  CUresult (*kernel_get_function_)(CUfunction* func, CUkernel kernel) = nullptr;

  CUresult (*launch_kernel_)(CUfunction f,
                             unsigned int grid_dim_x,
                             unsigned int grid_dim_y,
                             unsigned int grid_dim_z,
                             unsigned int block_dim_x,
                             unsigned int block_dim_y,
                             unsigned int block_dim_z,
                             unsigned int shared_mem_bytes,
                             CUstream stream,
                             void** kernel_params,
                             void** extra) = nullptr;

  CUresult (*library_load_data_)(CUlibrary* library,
                                 const void* code,
                                 CUjit_option* jit_options,
                                 void** jit_options_values,
                                 unsigned int num_jit_options,
                                 CUlibraryOption* library_options,
                                 void** library_option_values,
                                 unsigned int num_library_options)                       = nullptr;
  CUresult (*library_get_kernel_)(CUkernel* kernel, CUlibrary library, const char* name) = nullptr;
  CUresult (*library_unload_)(CUlibrary library)                                         = nullptr;

  CUresult (*mem_get_info_)(std::size_t* free, std::size_t* total) = nullptr;

#if LEGATE_DEFINED(LEGATE_CUDA_DRIVER_API_MOCK)
  FRIEND_TEST(ConfigureFBMemUnit, AutoConfigCUDA);
  FRIEND_TEST(ConfigureFBMemUnit, AutoConfigFail);
#endif
};

// ==========================================================================================

/**
 * @brief A RAII helper for managing the current *primary* context.
 *
 * On construction, it retains the primary context and pushes it as the current context. On
 * destruction it releases the context and pops it off the stack.
 */
class AutoPrimaryContext {
 public:
  AutoPrimaryContext(const AutoPrimaryContext&)            = delete;
  AutoPrimaryContext& operator=(const AutoPrimaryContext&) = delete;
  AutoPrimaryContext(AutoPrimaryContext&&)                 = delete;
  AutoPrimaryContext& operator=(AutoPrimaryContext&&)      = delete;

  /**
   * @brief Push the current primary context onto the stack.
   *
   * The current device is automatically detected based on the current processor. This may not
   * always provide the same answer for subsequent calls, so the user is highly encouraged to
   * use the `AutoPrimaryContext(CUdevice)` overload and pass a specific device argument.
   */
  AutoPrimaryContext();

  /**
   * @brief Push the current primary context onto the stack.
   *
   * @param device The device for which to push the primary context.
   */
  explicit AutoPrimaryContext(CUdevice device);

  /**
   * @brief Pop the current context and release the primary context.
   */
  ~AutoPrimaryContext();

 private:
  CUdevice device_{};
  CUcontext ctx_{};
};

// ==========================================================================================

/**
 * @brief A RAII helper for managing the current context.
 *
 * On construction, it pushes the context onto the CUDA context stack. On destruction it pops the
 * context off the stack.
 */
class AutoCUDAContext {
 public:
  AutoCUDAContext(const AutoCUDAContext&)            = delete;
  AutoCUDAContext& operator=(const AutoCUDAContext&) = delete;
  AutoCUDAContext(AutoCUDAContext&&)                 = delete;
  AutoCUDAContext& operator=(AutoCUDAContext&&)      = delete;

  /**
   * @brief Push the context onto the stack.
   *
   * @param ctx The context to push onto the stack.
   */
  explicit AutoCUDAContext(CUcontext ctx);

  /**
   * @brief Pop the context off the stack.
   */
  ~AutoCUDAContext();

 private:
  CUcontext ctx_{};
};

// ==========================================================================================

class LEGATE_EXPORT CUDADriverError : public std::runtime_error {
 public:
  CUDADriverError(const std::string& what, CUresult result);

  [[nodiscard]] CUresult error_code() const noexcept;

 private:
  CUresult result_{};
};

// ==========================================================================================

/**
 * @brief Set the current active CUDA driver API.
 *
 * This routine will set the current active CUDA driver returned by `get_cuda_driver_api()`. If
 * the new active driver would be at the same path as the current, it does nothing
 * (i.e. passing the same handle_path twice to this routine is a no-op).
 *
 * If the driver is replaced, the previous shared object is unloaded via a call to
 * `dlclose()`. If this causes the underlying shared object to be unloaded, any remaining
 * handles or function pointers originating from the shared object will become invalid.
 *
 * Therefore, the user is highly encouraged to store the return value of
 * `get_cuda_driver_api()` by value if the user intends to make multiple driver calls across a
 * "wide" gap.
 *
 * @param handle_path The path to, or name of, of the new driver shared object.
 */
void set_active_cuda_driver_api(std::string handle_path);

/**
 * @brief Get the singleton CUDA driver API object.
 *
 * The CUDA driver must have been set by a call to `set_active_cuda_driver_api()` prior to this
 * call, otherwise an exception is thrown. Usually this is done as part of `legate::start()`.
 *
 * @return The CUDA driver API.
 *
 * @throw std::runtime_error If the driver has not yet been set.
 */
[[nodiscard]] const InternalSharedPtr<CUDADriverAPI>& get_cuda_driver_api();

// ==========================================================================================

[[noreturn]] void throw_cuda_driver_error(CUresult result,
                                          std::string_view expression,
                                          std::string_view file,
                                          std::string_view func,
                                          int line);

}  // namespace legate::cuda::detail

#include <legate/cuda/detail/cuda_driver_api.inl>
