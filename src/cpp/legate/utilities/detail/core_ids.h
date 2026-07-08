/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

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
  IO_KVIKIO_FILE_READ,
  IO_KVIKIO_FILE_WRITE,
  IO_KVIKIO_TILE_READ,
  IO_KVIKIO_TILE_WRITE,
  IO_KVIKIO_TILE_BY_OFFSETS_READ,
  IO_HDF5_FILE_READ,
  IO_HDF5_FILE_WRITE_VDS,
  IO_HDF5_FILE_COMBINE_VDS,
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
enum class CoreTransform : std::int8_t {
  INVALID = -1,
  SHIFT   = 100,
  PROMOTE,
  PROJECT,
  BROADCAST,
  TRANSPOSE,
};
// NOLINTEND(readability-enum-initial-value)

enum class TaskPriority : std::int8_t { DEFAULT };

}  // namespace legate::detail
