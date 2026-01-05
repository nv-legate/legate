/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legion/api/types.h>

#include <cstdint>
#include <type_traits>

namespace legate {

enum class LocalTaskID : std::int64_t;
enum class LocalRedopID : std::int64_t;

}  // namespace legate

namespace legate::detail {

// You cannot convert enum class'es between each other via {} notation, that is:
//
// Enum2{Enum1::SOME_VALUE}
//
// does not compile. Instead you must static_cast(), but that's very tedious. So to get around
// this, we fake the enum class by making the enum "name" a namespace, and have enum itself be
// a regular (non class!) enum within it. That way we can easily do Type::VALUE and have the
// thing convert properly.
//
// Another alternative is to make the name into a struct, with an anonymous enum member, but
// this has several pitfalls:
//
// 1. sizeof(CoreTask) != sizeof(CoreTask::MAX_TASK). You can get around this by
//    adding a dummy padding member to the class to simulate the size, but that is hacky.
// 2. alignof(CoreTask) != alignof(CoreTask::MAX_TASK). Once again, a dummy padding member with
//    alignas(alignof(CoreTask::MAX_TASK)) works, but is a hack.
// 3. std::underlying_type does not work.
// 4. std::is_enum does not work.
//
// Using a namespace is preferable because for all of the issues above, the use of the
// namespace in place of the type (e.g. sizeof(Namespace), alignof(Namespace),
// std::is_enum_v<Namespace>), will result in a hard compilation failure. This ensure that
// these are caught and cannot silently produce wrong results.
namespace CoreTask {  // NOLINT(readability-identifier-naming)

// NOLINTBEGIN(readability-enum-initial-value)
enum CoreTask : std::underlying_type_t<LocalTaskID> {  // NOLINT(performance-enum-size)
  TOPLEVEL,
  EXTRACT_SCALAR,
  INIT_NCCL_ID,
  INIT_NCCL,
  FINALIZE_NCCL,
  INIT_CPUCOLL_MAPPING,
  INIT_CPUCOLL,
  FINALIZE_CPUCOLL,
  FIXUP_RANGES,
  OFFSETS_TO_RANGES,
  RANGES_TO_OFFSETS,
  FIND_BOUNDING_BOX,
  FIND_BOUNDING_BOX_SORTED,
  PREFETCH_BLOATED_INSTANCES,
  IO_KVIKIO_FILE_READ,
  IO_KVIKIO_FILE_WRITE,
  IO_KVIKIO_TILE_READ,
  IO_KVIKIO_TILE_WRITE,
  IO_KVIKIO_TILE_BY_OFFSETS_READ,
  IO_HDF5_FILE_READ,
  IO_HDF5_FILE_WRITE_VDS,
  IO_HDF5_FILE_COMBINE_VDS,
  OFFLOAD_TO,
  // NOTE: add core specific task IDs above FIRST_DYNAMIC_TASK
  FIRST_DYNAMIC_TASK,
  // Legate core runtime allocates MAX_TASK tasks from Legion upfront. All ID's prior to
  // FIRST_DYNAMIC_TASK are for specific, bespoke tasks. MAX_TASK - FIRST_DYNAMIC_TASK are for
  // "dynamic" tasks, e.g. those created from Python or elsewhere. Hence we make MAX_TASK large
  // enough so that there's enough slack.
  MAX_TASK = 1024,  // must be last
};

// NOLINTEND(readability-enum-initial-value)

}  // namespace CoreTask

// NOLINTBEGIN(readability-enum-initial-value)
enum class CoreProjectionOp : std::int32_t {
  // local id 0 always maps to the identity projection (global id 0)
  DELINEARIZE           = 2,
  FIRST_DYNAMIC_FUNCTOR = 10,
  MAX_FUNCTOR           = 3000000,
};
// NOLINTEND(readability-enum-initial-value)

// NOLINTBEGIN(readability-enum-initial-value)
enum class CoreShardID : std::underlying_type_t<CoreProjectionOp> {
  TOPLEVEL_TASK,
  LINEARIZE,
  // All sharding functors starting from CoreProjectionOp::FIRST_DYNAMIC_FUNCTOR should match
  // the projection functor of the same id. The sharding functor limit is thus the same as the
  // projection functor limit.
  MAX_FUNCTOR =
    static_cast<std::underlying_type_t<CoreProjectionOp>>(CoreProjectionOp::MAX_FUNCTOR),
};

// NOLINTEND(readability-enum-initial-value)

namespace CoreReductionOp {  // NOLINT(readability-identifier-naming)

enum CoreReductionOp : std::underlying_type_t<LocalRedopID> {  // NOLINT(performance-enum-size)
  JOIN_EXCEPTION,
  MAX_REDUCTION,
};

}  // namespace CoreReductionOp

enum class CoreSemanticTag : Legion::SemanticTag {  // NOLINT(performance-enum-size)
  // 0 is reserved by Legion for the object's name
  ALLOC_INFO = 1,
};

// NOLINTBEGIN(readability-enum-initial-value)
enum class CoreTransform : std::int8_t {
  INVALID = -1,
  SHIFT   = 100,
  PROMOTE,
  PROJECT,
  BROADCAST,
  TRANSPOSE,
  DELINEARIZE,
};
// NOLINTEND(readability-enum-initial-value)

enum class CoreMappingTag : Legion::MappingTagID {  // NOLINT(performance-enum-size)
  // 0 is reserved
  KEY_STORE = 1,
  MANUAL_PARALLEL_LAUNCH,
  TREE_REDUCE,
  JOIN_EXCEPTION,
};

enum class TaskPriority : std::int8_t { DEFAULT };

}  // namespace legate::detail
