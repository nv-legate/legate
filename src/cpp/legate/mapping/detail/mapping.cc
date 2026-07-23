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
