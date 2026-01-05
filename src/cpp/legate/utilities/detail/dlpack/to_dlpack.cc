/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/dlpack/to_dlpack.h>

#include <legate_defines.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/physical_store.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/dlpack/common.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/span.h>

#include <fmt/format.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace legate::detail {

namespace {

/**
 * @brief This class simply serves to wrap the physical store so that we keep the refcount
 * alive.
 *
 * A pointer to it is stuffed in the manager_ctx member.
 */
class ManagerContext {
 public:
  /**
   * @brief Construct the manager context.
   *
   * @param store The original physical store that should be kept alive.
   * @param data_deleter If the store's data needed to be copied (and therefore allocated),
   * this should be a function that knows how to delete that data.
   */
  explicit ManagerContext(legate::PhysicalStore store,
                          std::optional<std::function<void(void*)>> data_deleter);

  ManagerContext()                                 = delete;
  ManagerContext(const ManagerContext&)            = delete;
  ManagerContext& operator=(const ManagerContext&) = delete;
  ManagerContext(ManagerContext&&)                 = delete;
  ManagerContext& operator=(ManagerContext&&)      = delete;

  /**
   * @brief The function used as the deleter in `DLManagedTensorVersioned`.
   *
   * @param self The pointer to the managed tensor to delete.
   */
  static void delete_dlpack_ctx(DLManagedTensorVersioned* self);

 private:
  static void delete_dlpack_ctx_impl_(std::unique_ptr<DLManagedTensorVersioned> self);

  legate::PhysicalStore store_;
  std::optional<std::function<void(void*)>> data_deleter_{};
};

ManagerContext::ManagerContext(legate::PhysicalStore store,
                               std::optional<std::function<void(void*)>> data_deleter)
  : store_{std::move(store)}, data_deleter_{std::move(data_deleter)}
{
}

void ManagerContext::delete_dlpack_ctx(DLManagedTensorVersioned* self)
{
  delete_dlpack_ctx_impl_(std::unique_ptr<DLManagedTensorVersioned>{self});
}

// ------------------------------------------------------------------------------------------

void ManagerContext::delete_dlpack_ctx_impl_(std::unique_ptr<DLManagedTensorVersioned> self)
{
  // This deleter needs to handle these members being potentially NULL, because it is also used
  // as the deleter of the std::unique_ptr we return from to_dlpack(). So this deleter might be
  // called on a partially filled-out DLManagedTensorVersioned if any of the setup code of
  // to_dlpack() throws an exception.
  LEGATE_CHECK(self);
  LEGATE_CHECK(self->deleter == delete_dlpack_ctx || self->deleter == nullptr);
  // Carry this in a unique_ptr, since the deleters below are not guaranteed to be exception
  // safe. Well, that's not entirely true, we did write them ourselves after all, but it can
  // never hurt to be cautious.
  const auto holder_ptr = std::unique_ptr<ManagerContext>{
    static_cast<ManagerContext*>(std::exchange(self->manager_ctx, nullptr))};

  if (holder_ptr && holder_ptr->data_deleter_.has_value()) {
    if (auto* ptr = std::exchange(self->dl_tensor.data, nullptr)) {
      (*holder_ptr->data_deleter_)(ptr);
    }
  }
  delete[] std::exchange(self->dl_tensor.shape, nullptr);
  delete[] std::exchange(self->dl_tensor.strides, nullptr);
  self->deleter = nullptr;
}

// ==========================================================================================

constexpr bool operator<(const DLPackVersion& lhs, const DLPackVersion& rhs)
{
  return std::tie(lhs.major, lhs.minor) < std::tie(rhs.major, rhs.minor);
}

constexpr bool operator>(const DLPackVersion& lhs, const DLPackVersion& rhs)
{
  return std::tie(lhs.major, lhs.minor) > std::tie(rhs.major, rhs.minor);
}

/**
 * @brief Determine the version of the DLManagedTensor that we will produce.
 *
 * @param max_version The maximum version requested by the consumer. If not given, defaults to
 * whatever version of DLPack that Legate was built with.
 *
 * @return The DLPackVersion object.
 *
 * @throw std::runtime_error If we cannot accommodate the version request by the user.
 */
[[nodiscard]] DLPackVersion compute_version(const std::optional<DLPackVersion>& max_version)
{
  constexpr auto MAX_SUPPORTED_VERSION = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
  constexpr auto MIN_SUPPORTED_VERSION = DLPackVersion{1, 0};

  static_assert(MIN_SUPPORTED_VERSION < MAX_SUPPORTED_VERSION);
  static_assert(
    DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION == 1,  // NOLINT(misc-redundant-expression)
    "This DLPack code was only tested on version 1.1. Increase this manual version "
    "check when DLPack version is increased.");

  if (!max_version.has_value()) {
    return MAX_SUPPORTED_VERSION;
  }

  const auto& ver = *max_version;

  if (ver < MIN_SUPPORTED_VERSION) {
    throw TracedException<std::runtime_error>{fmt::format(
      "Cannot satisfy request for DLPack tensor of version {}.{}", ver.major, ver.minor)};
  }

  // If the user gives us max_version = 2.0, but we only provide 1.1, then it's still OK to
  // provide 1.1. Otherwise, we return whatever version they requested, which in this case can
  // only be 1.0 or 1.1.
  //
  // We can do this because version 1.1 added support for floating point vector types, which
  // legate does not support anyways (so from our POV, it is identical to 1.0). If we ever come
  // across a version which adds additional stuff, then we need to have a think about how to
  // refactor this code to support multiple versions.
  //
  // My suggestion would be to change this function to return a dispatcher function:
  //
  // - to_dlpack_v1_0()
  // - to_dlpack_v1_1()
  // - to_dlpack_v1_2()
  // - etc...
  //
  // Which then has the job of creating the DLManagedTensor:
  //
  // auto to_dlpack(...)
  // {
  //   ...
  //   const auto create_fn = compute_version(max_version);
  //
  //   return create_fn(store, stream, copy, ...);
  // }
  //
  // But cross that bridge when we get there.
  return ver > MAX_SUPPORTED_VERSION ? MAX_SUPPORTED_VERSION : ver;
}

/**
 * @brief Create the DLDevice descriptor from the corresponding Legate types.
 *
 * Physical stores cannot be "moved" to new memory, so `requested_device` is a bit useless. We
 * support it nonetheless in case the user passes it from Python, where we need to signal that
 * we cannot accommodate the movement.
 *
 * @param target The memory type of the Legate allocation.
 * @param requested_device The requested location from the consumer of where we should place
 * the store.
 *
 * @return The DLDevice object.
 *
 * @throw std::runtime_error If `requested_device` does not exactly match where the store is
 * already.
 */
[[nodiscard]] DLDevice make_device(mapping::StoreTarget target,
                                   const std::optional<DLDevice>& requested_device)
{
  auto ret = [target] {
    switch (target) {
      case mapping::StoreTarget::SYSMEM: [[fallthrough]];
      case mapping::StoreTarget::SOCKETMEM: {
        return DLDevice{/* device_type */ DLDeviceType::kDLCPU,
                        /* device_id */ 0};
      }
      case mapping::StoreTarget::FBMEM: {
        return DLDevice{/* device_type */ DLDeviceType::kDLCUDA,
                        /* device_id */ Runtime::get_runtime().get_current_cuda_device()};
      }
      case mapping::StoreTarget::ZCMEM: {
        return DLDevice{/* device_type */ DLDeviceType::kDLCUDAHost,
                        /* device_id */ Runtime::get_runtime().get_current_cuda_device()};
      }
    }
    LEGATE_ABORT("Unhandled allocation target ", target);
  }();

  if (requested_device.has_value()) {
    const auto [req_dtype, req_id] = *requested_device;
    const auto [dtype, id]         = ret;

    if (std::tie(req_dtype, req_id) != std::tie(dtype, id)) {
      throw TracedException<std::runtime_error>{
        fmt::format("Cannot satisfy request to provide DLPack tensor on device (device_type {}, "
                    "device_id {}). This task would provide a tensor on device (device_type {}, "
                    "device_id {}) instead.",
                    req_dtype,
                    req_id,
                    dtype,
                    id)};
    }
  }
  return ret;
}

/**
 * @brief Convert a Legate `Type` into the DLPack equivalent.
 *
 * @param type The Legate type.
 *
 * @return The DLDataType object.
 *
 * @throw std::domain_error If we cannot represent the byte-size of the Legate type in the
 * DLPack representation. This may happen, for example, if we have a very large binary type,
 * whose size exceeds the maximum value of std::uint8_t (which is used by DLPack to represent the
 * bit width).
 */
[[nodiscard]] DLDataType make_dtype(const legate::Type& type)
{
  constexpr auto MAX_BITS = std::numeric_limits<std::uint8_t>::max();
  const auto num_bits     = type.size() * CHAR_BIT;

  if (num_bits > MAX_BITS) {
    throw TracedException<std::domain_error>{
      fmt::format("Cannot convert Legate type {} to DLPack type. The number of bits required to "
                  "represent an element of the type ({}) > max size of DLPack bits datatype ({})",
                  type,
                  num_bits,
                  MAX_BITS)};
  }

  auto ret = DLDataType{};

  ret.code  = to_dlpack_type(type.code());
  ret.bits  = static_cast<std::uint8_t>(num_bits);
  ret.lanes = 1;  // We don't have vector types
  return ret;
}

/**
 * @brief Convert a Legate shape into the corresponding DLPack shape.
 *
 * @param domain The domain of the Legate store.
 *
 * @return A std::unique_ptr holding the DLPack shape.
 *
 * @throw std::invalid_argument If the dimension of the Legate store is negative.
 */
[[nodiscard]] std::unique_ptr<std::int64_t[]> make_shape(const Domain& domain)
{
  const auto dim = domain.get_dim();

  if (dim < 0) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Invalid domain dimension: {}, must be >=0", dim)};
  }
  // We still need to have *something* for 0-D, since we cannot guarantee that other
  // implementations handle a nullptr here properly
  if (dim == 0) {
    auto ret = std::make_unique<std::int64_t[]>(1);

    ret[0] = 0;
    return ret;
  }

  auto&& hi = domain.hi();
  auto&& lo = domain.lo();
  auto ret  = std::make_unique<std::int64_t[]>(static_cast<std::size_t>(dim));

  for (auto i = 0; i < dim; ++i) {
    ret[i] = std::max(hi[i] - lo[i] + 1, coord_t{0});
  }
  return ret;
}

/**
 * @brief Copy the strides of a Legate type to DLPack format.
 *
 * @param strides The strides to copy.
 * @param ty The type of the underlying Legate data.
 *
 * @return The DLPack strides.
 */
[[nodiscard]] std::unique_ptr<std::int64_t[]> copy_strides(Span<const std::size_t> strides,
                                                           const legate::Type& ty)
{
  const auto elem_size = ty.size();
  auto ret             = std::make_unique<std::int64_t[]>(strides.size());

  for (std::size_t i = 0; i < strides.size(); ++i) {
    // The strides are given in bytes, but DLPack wants them in elements
    ret[i] = static_cast<std::int64_t>(strides[i] / elem_size);
  }
  return ret;
}

/**
 * @brief Make a copy of the store data for DLPack.
 *
 * @param src The source array in bytes.
 * @param src_target The memory type of `src`.
 *
 * @return A std::unique_ptr holding the copied data. It also holds a deleter which knows how
 * to free the copied data.
 */
[[nodiscard]] std::unique_ptr<void, void (*)(void*)> copy_data(Span<const std::byte> src,
                                                               mapping::StoreTarget src_target)
{
  const auto num_bytes = src.size();

  switch (src_target) {
    case mapping::StoreTarget::FBMEM: [[fallthrough]];
    case mapping::StoreTarget::ZCMEM: {
      const auto ctx   = cuda::detail::AutoPrimaryContext{};
      auto&& api       = cuda::detail::get_cuda_driver_api();
      auto task_stream = Runtime::get_runtime().get_cuda_stream();

      // We use raw CUDA allocators here instead of Legion or Legate allocators because the
      // user has explicitly requested a full, standalone copy of the buffer. This buffer is
      // not subject to the standard memory management rules (e.g., task-based lifetime or
      // deferred deallocation in DeferredBuffer). Since the user can manage the buffer
      // independently -- potentially using it outside the context of Legate or Legion -- we
      // must ensure the memory is permanently allocated and not tied to any of our allocator's
      // lifetime.
      auto tmp = std::unique_ptr<void, void (*)(void*)>{
        api->mem_alloc_async(num_bytes, task_stream), [](void* ptr) {
          // Just have to hope that the deleter is called in a place where the current device
          // is the same as the one that allocated the pointer. I am not sure if CUDA
          // automatically handles this for us.
          const auto _ = cuda::detail::AutoPrimaryContext{};

          cuda::detail::get_cuda_driver_api()->mem_free_async(&ptr, LEGATE_CU_STREAM_DEFAULT);
        }};

      api->mem_cpy_async(tmp.get(), src.data(), num_bytes, task_stream);
      return tmp;
    }
    case mapping::StoreTarget::SYSMEM: [[fallthrough]];
    case mapping::StoreTarget::SOCKETMEM: {
      auto tmp = std::unique_ptr<void, void (*)(void*)>{
        new std::byte[num_bytes], [](void* ptr) { delete[] static_cast<std::byte*>(ptr); }};

      std::memcpy(tmp.get(), src.data(), num_bytes);
      return tmp;
    }
  }
  LEGATE_ABORT("Unhandled store target ", src_target);
}

/**
 * @brief Create the DLPack tensor object from a Legate physical store.
 *
 * @param store The store to create the tensor from.
 * @param device An optional device specifying the location where the data must end up.
 * @param must_copy Whether legate *must* make a copy of the data.
 *
 * @return A `std::pair` containing the produced tensor, and an optional deleter function to
 * delete the tensor's `data` member if `must_copy` was `true`.
 */
[[nodiscard]] std::pair<DLTensor, std::optional<std::function<void(void*)>>> make_tensor(
  const legate::PhysicalStore& store, const std::optional<DLDevice>& device, bool must_copy)
{
  const auto alloc = store.get_inline_allocation();
  const auto ty    = store.type();
  auto ret         = DLTensor{};

  ret.device      = make_device(alloc.target, device);
  ret.ndim        = store.dim();
  ret.dtype       = make_dtype(ty);
  ret.shape       = make_shape(store.domain()).release();
  ret.strides     = copy_strides(alloc.strides, ty).release();
  ret.byte_offset = 0;

  if (must_copy) {
    const auto num_bytes       = dl_tensor_size(ret);
    const auto* const as_bytes = static_cast<std::byte*>(alloc.ptr);
    auto copied                = copy_data({as_bytes, as_bytes + num_bytes}, alloc.target);

    ret.data = copied.release();
    return {std::move(ret), copied.get_deleter()};
  }

  ret.data = alloc.ptr;
  return {std::move(ret), std::nullopt};
}

/**
 * @brief Make the user stream wait on the current legate task stream.
 *
 * @param user_stream The stream to make wait.
 */
void wait_for_task_stream(CUstream user_stream)
{
  auto&& api = cuda::detail::get_cuda_driver_api();

  if (!api->is_loaded()) {
    throw TracedException<std::runtime_error>{
      fmt::format("Cannot wait for CUDA stream {}, because Legate {}.",
                  fmt::ptr(user_stream),
                  LEGATE_DEFINED(LEGATE_USE_CUDA) ? "failed to load the CUDA driver"
                                                  : "was not configured for CUDA")};
  }

  auto legate_stream = Runtime::get_runtime().get_cuda_stream();
  auto event         = api->event_create();

  try {
    api->event_record(event, legate_stream);
    api->stream_wait_event(user_stream, event);
  } catch (...) {
    api->event_destroy(&event);
    throw;
  }
  api->event_destroy(&event);
}

}  // namespace

std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)> to_dlpack(
  const legate::PhysicalStore& store,
  std::optional<bool> copy,
  std::optional<CUstream> stream,
  std::optional<DLPackVersion> max_version,
  std::optional<DLDevice> device)
{
  static_assert(
    DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION == 1,  // NOLINT(misc-redundant-expression)
    "This DLPack code was only tested on version 1.1. Increase this manual version "
    "check when DLPack version is increased.");

  using ret_type = std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>;

  const auto version   = compute_version(max_version);
  const auto must_copy = copy.value_or(false);
  auto data_deleter    = std::optional<std::function<void(void*)>>{};
  auto ret = ret_type{new DLManagedTensorVersioned{}, ManagerContext::delete_dlpack_ctx};

  ret->version = version;
  ret->deleter = ret.get_deleter();
  ret->flags   = 0;

  std::tie(ret->dl_tensor, data_deleter) = make_tensor(store, device, must_copy);

  if (must_copy) {
    ret->flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
    LEGATE_CHECK(data_deleter.has_value());
  }

  ret->manager_ctx = new ManagerContext{store, std::move(data_deleter)};

  if (stream.has_value()) {
    wait_for_task_stream(*stream);
  }
  return ret;
}

}  // namespace legate::detail
