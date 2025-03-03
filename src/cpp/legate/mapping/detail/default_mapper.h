/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>

#include <vector>

namespace legate::mapping::detail {

// FIXME: If clang-tidy ever lets us ignore warnings from specific headers, or fixes this, we
// can remove the NOLINT. This NOLINT is added to work around a bogus clang-tidy warning:
//
// _deps/legion-src/runtime/legion/legion_types.h:263:11: error: no definition found for
// 'DefaultMapper', but a definition with the same name 'DefaultMapper' found in another namespace
// 'legate::mapping::detail' [bugprone-forward-declaration-namespace,-warnings-as-errors]
//  263 |     class DefaultMapper;
//      |           ^
// legate/src/core/mapping/detail/default_mapper.h:21:7: note: a definition of
// 'DefaultMapper' is found here
//   21 | class DefaultMapper : public Mapper {
//      |       ^
//
// The only way (other than to disable the check wholesale), is to silence it for this class...
class DefaultMapper final : public Mapper {  // NOLINT(bugprone-forward-declaration-namespace)
 public:
  [[nodiscard]] std::vector<mapping::StoreMapping> store_mappings(
    const mapping::Task& task, const std::vector<StoreTarget>& options) override;
  [[nodiscard]] Scalar tunable_value(TunableID tunable_id) override;
  [[nodiscard]] std::optional<std::size_t> allocation_pool_size(const Task& task,
                                                                StoreTarget memory_kind) override;
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/default_mapper.inl>
