/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/experimental/io/detail/mapper.h>
#include <legate/mapping/operation.h>

namespace legate::experimental::io::detail {

std::vector<mapping::StoreMapping> Mapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  const auto target = options.front();
  // Require that all tasks get their Stores in contiguous buffers
  std::vector<mapping::StoreMapping> mappings;

  mappings.reserve(task.num_inputs() + task.num_outputs());
  for (std::size_t i = 0; i < task.num_inputs(); ++i) {
    auto&& store = task.input(static_cast<std::uint32_t>(i)).data();

    mappings.emplace_back(mapping::StoreMapping::default_mapping(store, target, /* exact */ true));
  }
  for (std::size_t i = 0; i < task.num_outputs(); ++i) {
    auto&& store = task.output(static_cast<std::uint32_t>(i)).data();

    mappings.emplace_back(mapping::StoreMapping::default_mapping(store, target, /* exact */ true));
  }
  return mappings;
}

legate::Scalar Mapper::tunable_value(TunableID)
{
  LEGATE_CHECK(false);
  return legate::Scalar{0};
}

}  // namespace legate::experimental::io::detail
