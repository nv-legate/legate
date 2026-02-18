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
#include <legate/utilities/dispatch.h>

#include <realm/memory.h>

#include <cstddef>
#include <vector>

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
 * @brief Returns the default Realm memory for the given storage target.
 *
 * Inline storage assumes single processor execution. As a result, return
 * the first available memory of the given type. This memory should be
 * the same memory for the default processor of the system.
 *
 * @param target The storage target to get the default Realm memory for.
 *
 * @return The default Realm memory for the given storage target.
 */
[[nodiscard]] Realm::Memory get_realm_memory(mapping::StoreTarget target)
{
  switch (target) {
    case mapping::StoreTarget::SYSMEM: [[fallthrough]];
    case mapping::StoreTarget::SOCKETMEM: {
      return Runtime::get_runtime().local_machine().system_memory();
    }
    case mapping::StoreTarget::FBMEM: {
      auto&& local_machine = Runtime::get_runtime().local_machine();
      const auto gpu       = local_machine.gpus().front();

      return local_machine.get_memory(gpu, target);
    }
    case mapping::StoreTarget::ZCMEM: {
      return Runtime::get_runtime().local_machine().zerocopy_memory();
    }
  }
  LEGATE_ABORT("Unhandled memory kind ", to_underlying(target));
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
      // For inline-storage, allocate memory on the first GPU
      const cuda::detail::AutoPrimaryContext ctx{0};
      auto&& api     = cuda::detail::get_cuda_driver_api();
      auto&& runtime = Runtime::get_runtime();

      return legate::ExternalAllocation::create_fbmem(
        static_cast<std::uint32_t>(runtime.get_current_cuda_device()),
        api->mem_alloc_async(size, runtime.get_cuda_stream()),
        size,
        /*read_only=*/false,
        [api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{0};

          api->mem_free_async(&ptr, Runtime::get_runtime().get_cuda_stream());
        });
    }
    case mapping::StoreTarget::ZCMEM: {
      // For inline-storage, allocate memory on the first GPU
      const cuda::detail::AutoPrimaryContext ctx{0};
      auto&& api = cuda::detail::get_cuda_driver_api();

      return legate::ExternalAllocation::create_zcmem(
        api->mem_alloc_managed(size),
        size,
        /*read_only=*/false,
        [api](void* ptr) {
          const cuda::detail::AutoPrimaryContext free_ctx{0};

          api->mem_free_async(&ptr, Runtime::get_runtime().get_cuda_stream());
        });
    }
  }
  LEGATE_ABORT("Unhandled memory kind ", to_underlying(target));
}

class MakeRegionInstance {
 public:
  template <std::int32_t DIM>
  [[nodiscard]] Realm::RegionInstance operator()(const Domain& domain,
                                                 std::uint32_t field_size,
                                                 const detail::ExternalAllocation& alloc) const
  {
    Realm::RegionInstance ret;

    // Later dimensions have smaller strides, following C-ordering.
    std::array<std::int32_t, DIM> dim_ordering;
    for (std::int32_t i = 0; i < DIM; ++i) {
      dim_ordering[i] = DIM - 1 - i;
    }

    auto* const layout = Realm::InstanceLayoutGeneric::choose_instance_layout<DIM, coord_t>(
      Realm::IndexSpace<DIM, coord_t>{static_cast<Rect<DIM, coord_t>>(domain)},
      Realm::InstanceLayoutConstraints{{field_size}, /*block_size=*/0},
      dim_ordering.data());

    // Modify the layout alignment accordingly to satisfy
    // the alignment of the provided allocation.
    const auto addr         = reinterpret_cast<std::int64_t>(alloc.ptr());
    std::uint32_t alignment = 1;

    while (addr % (alignment * 2) == 0) {
      alignment *= 2;
    }
    layout->alignment_reqd = alignment;

    LEGATE_CHECK(alloc.resource()->satisfies(*layout));

    const auto ready =
      Realm::RegionInstance::create_external_instance(ret,
                                                      get_realm_memory(alloc.target()),
                                                      *layout,
                                                      *alloc.resource(),
                                                      Realm::ProfilingRequestSet{});

    // Should always receive NO_EVENT because we are providing a valid external allocation
    // and we don't have any precondition.
    LEGATE_CHECK(ready == Realm::Event::NO_EVENT);
    LEGATE_CHECK(ret != Realm::RegionInstance::NO_INST);

    return ret;
  }
};

}  // namespace

InlineStorage::InlineStorage(const Domain& domain,
                             std::uint32_t field_size,
                             mapping::StoreTarget target)
  : domain_{domain},
    allocation_{make_external_alloc(target, domain.get_volume() * field_size)},
    region_instance_{
      dim_dispatch(domain.get_dim(), MakeRegionInstance{}, domain, field_size, *alloc_().impl())}
{
}

InlineStorage::InlineStorage(const Domain& domain,
                             std::uint32_t field_size,
                             const legate::ExternalAllocation& alloc)
  : domain_{domain},
    allocation_{alloc.impl()},
    region_instance_{
      dim_dispatch(domain.get_dim(), MakeRegionInstance{}, domain, field_size, *alloc_().impl())}
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

void* InlineStorage::data() { return alloc_().ptr(); }

const void* InlineStorage::data() const { return alloc_().ptr(); }

std::pair<Realm::RegionInstance, Realm::FieldID> InlineStorage::region_instance() const
{
  // Region instances from InlineStorage are created with a single field, so grab the first one.
  const Realm::FieldID fid = region_instance_.get_layout()->fields.begin()->first;
  return {region_instance_, fid};
}

void InlineStorage::remap_to(mapping::StoreTarget new_target)
{
  if (target() == new_target) {
    return;
  }

  const std::size_t volume = domain().get_volume();

  LEGATE_ASSERT(volume > 0);

  const auto num_bytes = alloc_().size();
  auto tmp = InlineStorage{domain(), static_cast<std::uint32_t>(num_bytes / volume), new_target};

  const auto device_copy = [&] {
    switch (new_target) {
      case mapping::StoreTarget::SYSMEM: [[fallthrough]];
      case mapping::StoreTarget::SOCKETMEM: {
        switch (target()) {
          case mapping::StoreTarget::SYSMEM: [[fallthrough]];
          case mapping::StoreTarget::SOCKETMEM: return false;  // Host to host
          case mapping::StoreTarget::FBMEM: [[fallthrough]];
          case mapping::StoreTarget::ZCMEM: return true;  // Device to host
        }
        LEGATE_ABORT("Unhandled target");
      }
      case mapping::StoreTarget::FBMEM: [[fallthrough]];
      case mapping::StoreTarget::ZCMEM: return true;  // Device to device or host to device
    }
    LEGATE_ABORT("Unhandled target");
  }();

  if (device_copy) {
    // Either source or destination is on device. This will handle either case transparently.
    const cuda::detail::AutoPrimaryContext ctx{0};

    cuda::detail::get_cuda_driver_api()->mem_cpy_async(
      tmp.data(), data(), num_bytes, Runtime::get_runtime().get_cuda_stream());

    // Wait so that copy is done prior to host memory being potentially freed.
    cuda::detail::stream_synchronize_minimal(Runtime::get_runtime().get_cuda_stream());
  } else {
    std::memcpy(tmp.data(), data(), num_bytes);
  }

  // detail::ExternalAllocation isn't properly RAII and won't delete the pointer on its own, so we
  // have to do it manually.
  alloc_().impl()->maybe_deallocate();
  *this = std::move(tmp);
}

}  // namespace legate::detail
