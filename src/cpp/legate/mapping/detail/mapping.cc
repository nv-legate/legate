/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/mapping.h>

#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
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

StoreTarget to_target(Memory::Kind kind)
{
  switch (kind) {
    case Memory::Kind::SYSTEM_MEM: return StoreTarget::SYSMEM;
    case Memory::Kind::GPU_FB_MEM: [[fallthrough]];
    case Memory::Kind::GPU_DYNAMIC_MEM: return StoreTarget::FBMEM;
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
    case Memory::Kind::GPU_MANAGED_MEM: break;
  }
  LEGATE_ABORT("Unhandled Memory::Kind ", legate::detail::to_underlying(kind));
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

legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> DimOrdering::dimension_order(
  std::uint32_t ndim) const
{
  const auto signed_ndim = static_cast<std::int32_t>(ndim);
  auto result            = legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{
    legate::detail::tags::size_tag, ndim, 0};

  switch (kind) {
    case Kind::C: {
      // The last dimension is the leading (unit-stride) one.
      std::generate(result.begin(), result.end(), [dim = signed_ndim]() mutable { return --dim; });
      return result;
    }
    case Kind::FORTRAN: {
      // The first dimension is the leading (unit-stride) one.
      std::iota(result.begin(), result.end(), 0);
      return result;
    }
    case Kind::CUSTOM: {
      if (dims.size() != ndim) {
        throw legate::detail::TracedException<std::invalid_argument>{
          fmt::format("Custom dimension ordering {} does not match the dimensionality {}",
                      fmt::join(dims, ", "),
                      ndim)};
      }

      std::unordered_set<std::int32_t> seen{};

      for (auto&& dim : dims) {
        if (dim < 0 || dim >= signed_ndim || !seen.insert(dim).second) {
          throw legate::detail::TracedException<std::invalid_argument>{
            fmt::format("Custom dimension ordering {} is not a permutation of [0, {})",
                        fmt::join(dims, ", "),
                        ndim)};
        }
      }
      std::copy(dims.begin(), dims.end(), result.begin());
      return result;
    }
  }
  LEGATE_ABORT("Unhandled DimOrdering::Kind ", legate::detail::to_underlying(kind));
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

}  // namespace legate::mapping::detail
