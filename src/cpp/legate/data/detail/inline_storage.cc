/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/inline_storage.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/allocation_cache.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace legate::detail {

namespace {

[[nodiscard]] CpuAllocationCache& get_cpu_allocation_cache()
{
  // 512 MB cache size for CPU allocations
  constexpr std::size_t CPU_CACHE_MAX_SIZE = 1U << 29U;

  // Use 32 byte alignment to increase opportunities
  // for vectorization, according to Realm
  constexpr std::size_t CPU_ALLOC_ALIGNMENT = 32;

  // Create a cache for CPU allocations to prevent
  // excessive page faults for larger, reusable allocations.
  static CpuAllocationCache cpu_allocation_cache{CPU_CACHE_MAX_SIZE, CPU_ALLOC_ALIGNMENT};

  return cpu_allocation_cache;
}

/**
 * @brief Allocates memory on the given storage target.
 *
 * The actual allocation is made on the default memory provided
 * by `get_realm_memory`.
 *
 * @param target The storage type to allocate memory on.
 * @param size The size of the memory to allocate.
 *
 * @return The external allocation of the allocated memory.
 */
[[nodiscard]] legate::ExternalAllocation make_external_alloc(mapping::StoreTarget target,
                                                             std::size_t size)
{
  switch (target) {
    case mapping::StoreTarget::SYSMEM: [[fallthrough]];
    case mapping::StoreTarget::SOCKETMEM: {
      void* new_ptr = get_cpu_allocation_cache().get_allocation(size);

      maybe_advise_huge_pages(new_ptr, size);

      return legate::ExternalAllocation::create_sysmem(
        new_ptr,
        size,
        /*read_only=*/false,
        [size](void* ptr) { get_cpu_allocation_cache().return_allocation(ptr, size); });
    }
    case mapping::StoreTarget::FBMEM: {
      auto&& runtime = Runtime::get_runtime();

      const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};
      auto&& api = cuda::detail::get_cuda_driver_api();

      return legate::ExternalAllocation::create_fbmem(
        static_cast<std::uint32_t>(runtime.get_current_cuda_device()),
        api->mem_alloc_async(size, runtime.get_cuda_stream()),
        size,
        /*read_only=*/false,
        [api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{
            has_finished() ? 0 : Runtime::get_runtime().get_current_cuda_device()};

          // Use null stream if the runtime has already finished.
          auto&& stream = has_finished() ? nullptr : Runtime::get_runtime().get_cuda_stream();

          api->mem_free_async(&ptr, stream);
        });
    }
    case mapping::StoreTarget::ZCMEM: {
      auto&& runtime = Runtime::get_runtime();

      // For inline-storage, allocate memory on the first GPU
      const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};
      auto&& api = cuda::detail::get_cuda_driver_api();

      return legate::ExternalAllocation::create_zcmem(
        api->mem_alloc_host(size),
        size,
        /*read_only=*/false,
        [api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{
            has_finished() ? 0 : Runtime::get_runtime().get_current_cuda_device()};

          api->mem_free_host(&ptr);
        });
    }
  }
  LEGATE_ABORT("Unhandled memory kind ", to_underlying(target));
}

}  // namespace

InlineStorage::InlineStorage(const Rect& domain,
                             std::uint32_t field_size,
                             mapping::StoreTarget target)
  : domain_{domain},
    field_size_{field_size},
    allocation_{
      make_external_alloc(target, std::max<>(domain.volume(), std::size_t{1}) * field_size)}
{
}

InlineStorage::InlineStorage(Rect domain,
                             std::uint32_t field_size,
                             const legate::ExternalAllocation& alloc)
  : domain_{std::move(domain)}, field_size_{field_size}, allocation_{alloc.impl()}
{
}

// ExternalAllocation doesn't delete the pointer in its destructor, so we have to do it
// manually.
InlineStorage::~InlineStorage()
{
  if (alloc_().impl()) {
    alloc_().impl()->maybe_deallocate();
  }
}

mapping::StoreTarget InlineStorage::target() const { return alloc_().target(); }

void InlineStorage::rebase(const Rect& domain)
{
  LEGATE_ASSERT(domain.volume() == domain_.volume());
  domain_ = domain;
}

void* InlineStorage::data() { return alloc_().ptr(); }

const void* InlineStorage::data() const { return alloc_().ptr(); }

std::pair<void*, Point> InlineStorage::base_and_element_strides() const
{
  auto element_strides = domain_.strides();
  auto* const base =
    static_cast<std::int8_t*>(alloc_().ptr()) - (domain_.lo().dot(element_strides) * field_size_);

  return {static_cast<void*>(base), std::move(element_strides)};
}

}  // namespace legate::detail
