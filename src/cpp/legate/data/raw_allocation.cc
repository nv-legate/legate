/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/raw_allocation.h>

#include <legate_defines.h>

#include <legate/data/detail/raw_allocation.h>
#include <legate/mapping/detail/machine.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate {

bool RawAllocation::read_only() const { return impl_->read_only(); }

mapping::StoreTarget RawAllocation::target() const { return impl_->target(); }

void* RawAllocation::ptr() const { return impl_->ptr(); }

std::size_t RawAllocation::size() const { return impl_->size(); }

RawAllocation::~RawAllocation() noexcept = default;

namespace {

void check_non_null(const void* ptr)
{
  if (!ptr) {
    throw detail::TracedException<std::invalid_argument>{
      "Raw allocation cannot be created from a null pointer"};
  }
}

}  // namespace

/*static*/ RawAllocation RawAllocation::create_sysmem(
  void* ptr,
  std::size_t size,
  bool read_only /*=true*/,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
  check_non_null(ptr);
  return RawAllocation{make_internal_shared<detail::RawAllocation>(
    read_only, mapping::StoreTarget::SYSMEM, ptr, size, std::move(deleter))};
}

/*static*/ RawAllocation RawAllocation::create_zcmem(
  void* ptr,
  std::size_t size,
  bool read_only,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
  if (!LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    throw detail::TracedException<std::runtime_error>{"CUDA support is unavailable"};
  }

  check_non_null(ptr);
  return RawAllocation{make_internal_shared<detail::RawAllocation>(
    read_only, mapping::StoreTarget::ZCMEM, ptr, size, std::move(deleter))};
}

/*static*/ RawAllocation RawAllocation::create_fbmem(
  std::uint32_t local_device_id,
  void* ptr,
  std::size_t size,
  bool read_only,
  std::optional<Deleter> deleter /*=std::nullopt*/)  // NOLINT(performance-unnecessary-value-param)
{
  if (!LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    throw detail::TracedException<std::runtime_error>{"CUDA support is unavailable"};
  }

  if (const auto num_gpus = detail::Runtime::get_runtime().local_machine().total_gpu_count();
      local_device_id >= num_gpus) {
    throw detail::TracedException<std::out_of_range>{fmt::format(
      "Invalid local device ID {}: this rank has {} GPU(s)", local_device_id, num_gpus)};
  }
  check_non_null(ptr);
  // TODO(wonchanl): Need to handle the case where local_device_id is not the device managed by this
  // rank
  return RawAllocation{make_internal_shared<detail::RawAllocation>(
    read_only, mapping::StoreTarget::FBMEM, ptr, size, std::move(deleter))};
}

/*static*/ RawAllocation RawAllocation::create_sysmem(
  const void* ptr, std::size_t size, std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_sysmem(const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

/*static*/ RawAllocation RawAllocation::create_zcmem(
  const void* ptr, std::size_t size, std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_zcmem(const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

/*static*/ RawAllocation RawAllocation::create_fbmem(
  std::uint32_t local_device_id,
  const void* ptr,
  std::size_t size,
  std::optional<Deleter> deleter /*=std::nullopt*/)
{
  return create_fbmem(
    local_device_id, const_cast<void*>(ptr), size, /*read_only=*/true, std::move(deleter));
}

}  // namespace legate
