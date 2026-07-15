/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/remap_request.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace legate::detail {

namespace {

[[nodiscard]] bool is_device_target(mapping::StoreTarget target)
{
  return target == mapping::StoreTarget::FBMEM || target == mapping::StoreTarget::ZCMEM;
}

}  // namespace

bool RemapRequest::requires_host_sync() const
{
  // Only a device-to-host copy needs to be awaited before the host reads the destination.
  return !is_device_target(target_->target()) && is_device_target(source_->target());
}

void RemapRequest::perform(VariantCode /*variant_code*/) const
{
  const std::size_t volume    = std::max<>(std::size_t{1}, source_->domain().get_volume());
  const std::size_t num_bytes = volume * field_size_;

  if (is_device_target(source_->target()) || is_device_target(target_->target())) {
    // Either the source or the destination is on the device; a single asynchronous copy handles
    // every direction (device-to-host, host-to-device, and device-to-device) transparently.
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    auto&& api = cuda::detail::get_cuda_driver_api();

    api->mem_cpy_async(
      target_->data(), source_->data(), num_bytes, Runtime::get_runtime().get_cuda_stream());
  } else {
    std::memcpy(target_->data(), source_->data(), num_bytes);
  }
}

}  // namespace legate::detail
