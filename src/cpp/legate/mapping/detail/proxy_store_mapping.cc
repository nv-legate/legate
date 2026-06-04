/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/proxy_store_mapping.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/task.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <variant>

namespace legate::mapping::detail {

ProxyStoreMapping::ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                                  ProxyInputArguments,
                                                  ProxyOutputArguments,
                                                  ProxyReductionArguments> store,
                                     ProxyInstanceMappingPolicy policy)
  : stores_{std::move(store)}, policy_{std::move(policy)}
{
}

void ProxyStoreMapping::apply_legion(const Task&,
                                     Span<const StoreTarget>,
                                     std::vector<mapping::StoreMapping>*) const
{
}

void ProxyStoreMapping::apply_inline(const legate::detail::TaskBase&,
                                     Span<const StoreTarget>,
                                     legate::detail::SmallVector<InstanceMappingPolicy>*,
                                     legate::detail::SmallVector<InstanceMappingPolicy>*,
                                     legate::detail::SmallVector<InstanceMappingPolicy>*) const
{
}

bool operator==(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs)
{
  if (std::addressof(lhs) == std::addressof(rhs)) {
    return true;
  }

  return std::tie(lhs.stores(), lhs.policy()) == std::tie(rhs.stores(), rhs.policy());
}

bool operator!=(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs)
{
  return !(lhs == rhs);
}

}  // namespace legate::mapping::detail
