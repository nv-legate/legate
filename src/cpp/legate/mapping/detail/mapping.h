/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace legate::mapping::detail {

/**
 * @brief Convert a VariantCode to its corresponding TaskTarget.
 *
 * @param code The variant code to convert.
 *
 * @return The TaskTarget associated with the given VariantCode.
 */
[[nodiscard]] TaskTarget to_target(VariantCode code);

[[nodiscard]] TaskTarget to_target(Processor::Kind kind);

[[nodiscard]] TaskTarget get_matching_task_target(StoreTarget target);

[[nodiscard]] StoreTarget to_target(Memory::Kind kind);

[[nodiscard]] Processor::Kind to_kind(TaskTarget target);

[[nodiscard]] Processor::Kind to_kind(VariantCode code);

[[nodiscard]] Memory::Kind to_kind(StoreTarget target);

[[nodiscard]] VariantCode to_variant_code(TaskTarget target);

[[nodiscard]] VariantCode to_variant_code(Processor::Kind kind);

class DimOrdering {
 public:
  using Kind = mapping::DimOrdering::Kind;

  explicit DimOrdering(Kind _kind);
  explicit DimOrdering(std::vector<std::int32_t> _dims);

  [[nodiscard]] bool operator==(const DimOrdering& other) const;

  Kind kind{};
  std::vector<std::int32_t> dims{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/mapping.inl>
