/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/mapping.h>

#include <legate/utilities/detail/type_traits.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>

namespace legate::mapping::detail {

TaskTarget to_target(VariantCode code)
{
  switch (code) {
    case VariantCode::GPU: return TaskTarget::GPU;
    case VariantCode::OMP: return TaskTarget::OMP;
    case VariantCode::CPU: return TaskTarget::CPU;
  }
  LEGATE_ABORT("Unhandled VariantCode ", legate::detail::to_underlying(code));
}

TaskTarget to_target(Processor::Kind kind)
{
  switch (kind) {
    case Processor::Kind::TOC_PROC: return TaskTarget::GPU;
    case Processor::Kind::OMP_PROC: return TaskTarget::OMP;
    case Processor::Kind::LOC_PROC:
      [[fallthrough]];
      // Note, returning TaskTarget::CPU for Processor::Kind::PY_PROC isn't technically
      // correct. This is "safe" to do because the only caller that might pass PY_PROC is the
      // inline task launch.
    case Processor::Kind::PY_PROC: return TaskTarget::CPU;
    case Processor::Kind::NO_KIND: [[fallthrough]];
    case Processor::Kind::UTIL_PROC: [[fallthrough]];
    case Processor::Kind::IO_PROC: [[fallthrough]];
    case Processor::Kind::PROC_GROUP: [[fallthrough]];
    case Processor::Kind::PROC_SET: break;
  }
  LEGATE_ABORT("Unhandled Processor::Kind ", legate::detail::to_underlying(kind));
}

TaskTarget get_matching_task_target(StoreTarget target)
{
  switch (target) {
    case StoreTarget::ZCMEM: [[fallthrough]];
    case StoreTarget::FBMEM: return TaskTarget::GPU;
    case StoreTarget::SOCKETMEM: return TaskTarget::OMP;
    case StoreTarget::SYSMEM: return TaskTarget::CPU;
  }
  LEGATE_ABORT("Unhandled StoreTarget: ", target);
}

StoreTarget to_target(Memory::Kind kind)
{
  switch (kind) {
    case Memory::Kind::SYSTEM_MEM: return StoreTarget::SYSMEM;
    case Memory::Kind::GPU_FB_MEM: return StoreTarget::FBMEM;
    case Memory::Kind::Z_COPY_MEM: return StoreTarget::ZCMEM;
    case Memory::Kind::SOCKET_MEM: return StoreTarget::SOCKETMEM;
    case Memory::Kind::NO_MEMKIND: [[fallthrough]];
    case Memory::Kind::GLOBAL_MEM: [[fallthrough]];
    case Memory::Kind::REGDMA_MEM: [[fallthrough]];
    case Memory::Kind::DISK_MEM: [[fallthrough]];
    case Memory::Kind::HDF_MEM: [[fallthrough]];
    case Memory::Kind::FILE_MEM: [[fallthrough]];
    case Memory::Kind::LEVEL3_CACHE: [[fallthrough]];
    case Memory::Kind::LEVEL2_CACHE: [[fallthrough]];
    case Memory::Kind::LEVEL1_CACHE: [[fallthrough]];
    case Memory::Kind::GPU_MANAGED_MEM: [[fallthrough]];
    case Memory::Kind::GPU_DYNAMIC_MEM: break;
  }
  LEGATE_ABORT("Unhandled Processor::Kind ", legate::detail::to_underlying(kind));
}

Processor::Kind to_kind(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return Processor::Kind::TOC_PROC;
    case TaskTarget::OMP: return Processor::Kind::OMP_PROC;
    case TaskTarget::CPU: return Processor::Kind::LOC_PROC;
  }
  LEGATE_ABORT("Unhandled TaskTarget ", legate::detail::to_underlying(target));
}

Processor::Kind to_kind(VariantCode code)
{
  switch (code) {
    case VariantCode::CPU: return Processor::Kind::LOC_PROC;
    case VariantCode::GPU: return Processor::Kind::TOC_PROC;
    case VariantCode::OMP: return Processor::Kind::OMP_PROC;
  }
  LEGATE_ABORT("Unhandled variant code ", legate::detail::to_underlying(code));
}

Memory::Kind to_kind(StoreTarget target)
{
  switch (target) {
    case StoreTarget::SYSMEM: return Memory::Kind::SYSTEM_MEM;
    case StoreTarget::FBMEM: return Memory::Kind::GPU_FB_MEM;
    case StoreTarget::ZCMEM: return Memory::Kind::Z_COPY_MEM;
    case StoreTarget::SOCKETMEM: return Memory::Kind::SOCKET_MEM;
  }
  LEGATE_ABORT("Unhandled StoreTarget ", legate::detail::to_underlying(target));
}

VariantCode to_variant_code(TaskTarget target)
{
  switch (target) {
    case TaskTarget::GPU: return VariantCode::GPU;
    case TaskTarget::OMP: return VariantCode::OMP;
    case TaskTarget::CPU: return VariantCode::CPU;
  }
  LEGATE_ABORT("Unhandled TaskTarget ", legate::detail::to_underlying(target));
}

VariantCode to_variant_code(Processor::Kind kind) { return to_variant_code(to_target(kind)); }

// ==========================================================================================

std::vector<Legion::DimensionKind> DimOrdering::generate_legion_dims(std::uint32_t ndim) const
{
  std::vector<Legion::DimensionKind> ordering;

  LEGATE_CHECK(ndim > 0);
  ordering.reserve(static_cast<std::size_t>(ndim) + 1);
  switch (kind) {
    case Kind::C: {
      for (auto dim = static_cast<std::int32_t>(ndim) - 1; dim >= 0; --dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::FORTRAN: {
      for (std::uint32_t dim = 0; dim < ndim; ++dim) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
    case Kind::CUSTOM: {
      LEGATE_CHECK(ndim == dims.size());
      for (auto dim : dims) {
        ordering.push_back(static_cast<Legion::DimensionKind>(LEGION_DIM_X + dim));
      }
      break;
    }
  }
  // After listing all dimension names, we add the field dimension as the last one so the instance
  // will have an SOA layout
  ordering.push_back(LEGION_DIM_F);

  return ordering;
}

}  // namespace legate::mapping::detail
