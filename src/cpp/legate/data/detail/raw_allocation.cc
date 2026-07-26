/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/raw_allocation.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/allocation_cache.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/type_traits.h>

#include <cstddef>

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

}  // namespace

RawAllocation::~RawAllocation() { maybe_deallocate(); }

void RawAllocation::maybe_deallocate()
{
  if (deleter_) {
    (*deleter_)(ptr_);
    deleter_.reset();
  }
}

InternalSharedPtr<RawAllocation> RawAllocation::create(mapping::StoreTarget target,
                                                       std::size_t size)
{
  switch (target) {
    case mapping::StoreTarget::SYSMEM: [[fallthrough]];
    case mapping::StoreTarget::SOCKETMEM: {
      void* new_ptr = get_cpu_allocation_cache().get_allocation(size);

      maybe_advise_huge_pages(new_ptr, size);

      return make_internal_shared<RawAllocation>(
        /*read_only=*/false,
        mapping::StoreTarget::SYSMEM,
        new_ptr,
        size,
        RawAllocation::Deleter{
          [size](void* ptr) { get_cpu_allocation_cache().return_allocation(ptr, size); }});
    }
    case mapping::StoreTarget::FBMEM: {
      auto&& runtime = Runtime::get_runtime();

      const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};
      auto&& api = cuda::detail::get_cuda_driver_api();

      return make_internal_shared<RawAllocation>(
        /*read_only=*/false,
        mapping::StoreTarget::FBMEM,
        api->mem_alloc_async(size, runtime.get_cuda_stream()),
        size,
        RawAllocation::Deleter{[api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{
            has_finished() ? 0 : Runtime::get_runtime().get_current_cuda_device()};

          // Use null stream if the runtime has already finished.
          auto&& stream = has_finished() ? nullptr : Runtime::get_runtime().get_cuda_stream();

          api->mem_free_async(&ptr, stream);
        }});
    }
    case mapping::StoreTarget::ZCMEM: {
      auto&& runtime = Runtime::get_runtime();

      // For inline-storage, allocate memory on the first GPU
      const cuda::detail::AutoPrimaryContext ctx{runtime.get_current_cuda_device()};
      auto&& api = cuda::detail::get_cuda_driver_api();

      return make_internal_shared<RawAllocation>(
        /*read_only=*/false,
        mapping::StoreTarget::ZCMEM,
        api->mem_alloc_host(size),
        size,
        RawAllocation::Deleter{[api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{
            has_finished() ? 0 : Runtime::get_runtime().get_current_cuda_device()};

          api->mem_free_host(&ptr);
        }});
    }
  }
  LEGATE_ABORT("Unhandled memory kind ", to_underlying(target));
  return nullptr;
}

}  // namespace legate::detail
