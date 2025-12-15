/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/array.h>
#include <legate/mapping/detail/store.h>
#include <legate/utilities/detail/tuple.h>

namespace mapping_utils_test {

// Helper function to create a test store using FutureWrapper
[[nodiscard]] inline legate::mapping::detail::Store create_test_store(const legate::Shape& shape,
                                                                      const legate::Type& type)
{
  // Get internal type from public Type - use InternalSharedPtr constructor from SharedPtr
  legate::InternalSharedPtr<legate::detail::Type> internal_type{type.impl()};

  // Create domain from shape using to_domain() helper
  const auto& extents = shape.extents();
  std::vector<std::uint64_t> shape_vec(extents.begin(), extents.end());
  const Legion::Domain domain = legate::detail::to_domain(shape_vec);

  auto future_wrapper = legate::mapping::detail::FutureWrapper{/*idx=*/0, domain};

  // Create an empty (identity) TransformStack to avoid nullptr dereference in Store::domain()
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();

  return legate::mapping::detail::Store{static_cast<std::int32_t>(shape.extents().size()),
                                        std::move(internal_type),
                                        std::move(future_wrapper),
                                        std::move(transform)};
}

}  // namespace mapping_utils_test
