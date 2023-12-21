/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/data/external_allocation.h"

#include "core/data/detail/external_allocation.h"
#include "core/runtime/detail/runtime.h"

#include "legate_defines.h"
#include "realm/instance.h"

#if LegateDefined(LEGATE_USE_CUDA)
#include "realm/cuda/cuda_access.h"
#endif

#include <memory>
#include <stdexcept>

namespace legate {

// TODO: For some reason having these two methods in the .inl file leads to a missing symbol
// error.
ExternalAllocation::ExternalAllocation(InternalSharedPtr<detail::ExternalAllocation>&& impl)
  : impl_{std::move(impl)}
{
}

const SharedPtr<detail::ExternalAllocation>& ExternalAllocation::impl() const { return impl_; }

bool ExternalAllocation::read_only() const { return impl_->read_only(); }

mapping::StoreTarget ExternalAllocation::target() const { return impl_->target(); }

void* ExternalAllocation::ptr() const { return impl_->ptr(); }

size_t ExternalAllocation::size() const { return impl_->size(); }

ExternalAllocation::~ExternalAllocation() noexcept = default;

/*static*/ ExternalAllocation ExternalAllocation::create_sysmem(
  void* ptr,
  size_t size,
  bool read_only /*=true*/,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
  auto realm_resource = std::make_unique<Realm::ExternalMemoryResource>(
    reinterpret_cast<uintptr_t>(ptr), size, read_only);
  return ExternalAllocation{
    make_internal_shared<detail::ExternalAllocation>(read_only,
                                                     mapping::StoreTarget::SYSMEM,
                                                     ptr,
                                                     size,
                                                     std::move(realm_resource),
                                                     std::move(deleter))};
}

/*static*/ ExternalAllocation ExternalAllocation::create_zcmem(
  void* ptr,
  size_t size,
  bool read_only,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
#if LegateDefined(LEGATE_USE_CUDA)
  auto realm_resource = std::make_unique<Realm::ExternalCudaPinnedHostResource>(
    reinterpret_cast<uintptr_t>(ptr), size, read_only);
  return ExternalAllocation{
    make_internal_shared<detail::ExternalAllocation>(read_only,
                                                     mapping::StoreTarget::ZCMEM,
                                                     ptr,
                                                     size,
                                                     std::move(realm_resource),
                                                     std::move(deleter))};
#else
  static_cast<void>(ptr);
  static_cast<void>(size);
  static_cast<void>(read_only);
  static_cast<void>(deleter);
  throw std::runtime_error{"CUDA support is unavailable"};
  return {};
#endif
}

/*static*/ ExternalAllocation ExternalAllocation::create_fbmem(
  uint32_t local_device_id,
  void* ptr,
  size_t size,
  bool read_only,
  std::optional<ExternalAllocation::Deleter> deleter /*=std::nullopt*/)
{
#if LegateDefined(LEGATE_USE_CUDA)
  auto& local_gpus = detail::Runtime::get_runtime()->local_machine().gpus();
  if (local_device_id >= local_gpus.size()) {
    throw std::out_of_range{"Device ID " + std::to_string(local_device_id) +
                            " is invalid (the runtime is configured " + "with only " +
                            std::to_string(local_gpus.size())};
  }

  auto realm_resource = std::make_unique<Realm::ExternalCudaMemoryResource>(
    local_device_id, reinterpret_cast<uintptr_t>(ptr), size, read_only);
  return ExternalAllocation{
    make_internal_shared<detail::ExternalAllocation>(read_only,
                                                     mapping::StoreTarget::FBMEM,
                                                     ptr,
                                                     size,
                                                     std::move(realm_resource),
                                                     std::move(deleter))};
#else
  static_cast<void>(local_device_id);
  static_cast<void>(ptr);
  static_cast<void>(size);
  static_cast<void>(read_only);
  static_cast<void>(deleter);
  throw std::runtime_error{"CUDA support is unavailable"};
  return {};
#endif
}

}  // namespace legate
