/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/store_mapping_signature.h>

#include <legate/mapping/detail/operation.h>
#include <legate/mapping/detail/proxy_store_mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/mapping/operation.h>
#include <legate/mapping/proxy_store_mapping.h>
#include <legate/utilities/detail/zip.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace legate::detail {

std::vector<mapping::StoreMapping> StoreMappingSignature::apply(
  const mapping::detail::Task& task, Span<const mapping::StoreTarget> options) const
{
  auto ret = std::vector<mapping::StoreMapping>{};

  ret.reserve(store_mappings().size());
  for (auto&& proxy : store_mappings()) {
    proxy->apply(task, options, &ret);
  }
  return ret;
}

bool operator==(const StoreMappingSignature& lhs, const StoreMappingSignature& rhs)
{
  if (std::addressof(lhs) == std::addressof(rhs)) {
    return true;
  }

  // Must do this check first, zip_equal will throw if the input containers are not of equal
  // size
  if (lhs.store_mappings().size() != rhs.store_mappings().size()) {
    return false;
  }

  const auto zipper = zip_equal(lhs.store_mappings(), rhs.store_mappings());

  return std::all_of(zipper.begin(), zipper.end(), [](const auto& tuple) {
    auto&& [lhs_v, rhs_v] = tuple;

    return *lhs_v == *rhs_v;
  });
}

bool operator!=(const StoreMappingSignature& lhs, const StoreMappingSignature& rhs)
{
  return !(lhs == rhs);
}

}  // namespace legate::detail
