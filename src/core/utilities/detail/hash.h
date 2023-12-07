/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/utilities/hash.h"
#include "core/utilities/typedefs.h"

#include "legion.h"

namespace legate {

template <typename T1, typename T2>
struct hasher<std::pair<T1, T2>> {
  [[nodiscard]] size_t operator()(const std::pair<T1, T2>& v) const noexcept
  {
    return hash_all(v.first, v.second);
  }
};

}  // namespace legate

namespace std {

template <>
struct hash<Legion::IndexSpace> {
  [[nodiscard]] size_t operator()(const Legion::IndexSpace& index_space) const noexcept
  {
    return legate::hash_all(index_space.get_id(), index_space.get_tree_id());
  }
};

template <>
struct hash<Legion::IndexPartition> {
  [[nodiscard]] size_t operator()(const Legion::IndexPartition& partition) const noexcept
  {
    return legate::hash_all(partition.get_id(), partition.get_tree_id());
  }
};

template <>
struct hash<Legion::FieldSpace> {
  [[nodiscard]] size_t operator()(const Legion::FieldSpace& field_space) const noexcept
  {
    return legate::hash_all(field_space.get_id());
  }
};

template <>
struct hash<Legion::LogicalRegion> {
  [[nodiscard]] size_t operator()(const Legion::LogicalRegion& region) const noexcept
  {
    // tree ids uniquely identify region tress, so no need to hash field spaces here
    return legate::hash_all(region.get_index_space(), region.get_tree_id());
  }
};

template <>
struct hash<Legion::LogicalPartition> {
  [[nodiscard]] size_t operator()(const Legion::LogicalPartition& partition) const noexcept
  {
    return legate::hash_all(partition.get_index_partition(), partition.get_tree_id());
  }
};

template <>
struct hash<legate::Domain> {
  [[nodiscard]] size_t operator()(const legate::Domain& domain) const noexcept
  {
    size_t result = 0;
    for (int32_t idx = 0; idx < 2 * domain.dim; ++idx) {
      legate::hash_combine(result, domain.rect_data[idx]);
    }
    return result;
  }
};

}  // namespace std
