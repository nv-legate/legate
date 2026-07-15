/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/external_allocation.h>

#include <legate_defines.h>

#include <legate/data/detail/external_allocation.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate {

bool ExternalAllocation::read_only() const { return impl_->read_only(); }

mapping::StoreTarget ExternalAllocation::target() const { return impl_->target(); }

void* ExternalAllocation::ptr() const { return impl_->ptr(); }

std::size_t ExternalAllocation::size() const { return impl_->size(); }

ExternalAllocation::~ExternalAllocation() noexcept = default;

/*static*/ ExternalAllocation ExternalAllocation::create_sysmem(
  void* ptr,
  std::size_t size,
  bool read_only /*=true*/,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
  return ExternalAllocation{make_internal_shared<detail::ExternalAllocation>(
    read_only, mapping::StoreTarget::SYSMEM, ptr, size, std::move(deleter))};
}

/*static*/ ExternalAllocation ExternalAllocation::create_zcmem(
  void* ptr,
  std::size_t size,
  bool read_only,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  return ExternalAllocation{make_internal_shared<detail::ExternalAllocation>(
    read_only, mapping::StoreTarget::ZCMEM, ptr, size, std::move(deleter))};
#else
  static_cast<void>(ptr);
  static_cast<void>(size);
  static_cast<void>(read_only);
  static_cast<void>(deleter);
  throw detail::TracedException<std::runtime_error>{"CUDA support is unavailable"};
  return {};
#endif
}

/*static*/ ExternalAllocation ExternalAllocation::create_fbmem(
  std::uint32_t local_device_id,
  void* ptr,
  std::size_t size,
  bool read_only,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static_cast<void>(local_device_id);
  return ExternalAllocation{make_internal_shared<detail::ExternalAllocation>(
    read_only, mapping::StoreTarget::FBMEM, ptr, size, std::move(deleter))};
#else
  static_cast<void>(local_device_id);
  static_cast<void>(ptr);
  static_cast<void>(size);
  static_cast<void>(read_only);
  static_cast<void>(deleter);
  throw detail::TracedException<std::runtime_error>{"CUDA support is unavailable"};
  return {};
#endif
}

/*static*/ ExternalAllocation ExternalAllocation::create_sysmem(
  const void* ptr, std::size_t size, std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_sysmem(const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

/*static*/ ExternalAllocation ExternalAllocation::create_zcmem(
  const void* ptr, std::size_t size, std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_zcmem(const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

/*static*/ ExternalAllocation ExternalAllocation::create_fbmem(
  std::uint32_t local_device_id,
  const void* ptr,
  std::size_t size,
  std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_fbmem(
    local_device_id, const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

}  // namespace legate
