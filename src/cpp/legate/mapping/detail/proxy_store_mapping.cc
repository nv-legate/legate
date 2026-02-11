/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/proxy_store_mapping.h>

#include <legate/mapping/detail/array.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/detail/operation.h>
#include <legate/mapping/detail/store.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/store_iterator_cache.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

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

namespace {

void populate_store_mappings(const InstanceMappingPolicy& policy,
                             Span<const InternalSharedPtr<Array>> args,
                             std::vector<mapping::StoreMapping>* store_mappings)
{
  auto cache = legate::detail::StoreIteratorCache<InternalSharedPtr<Store>>{};

  for (auto&& arr : args) {
    auto&& stores = cache(*arr);

    store_mappings->emplace_back(std::make_unique<StoreMapping>(policy, stores));
  }
}

}  // namespace

void ProxyStoreMapping::apply(const Task& task,
                              Span<const StoreTarget> options,
                              std::vector<mapping::StoreMapping>* store_mappings) const
{
  const auto concrete_policy = [&] {
    auto&& p = policy();

    // Do not use the factory functions for this constructor (e.g. `with_ordering()`,
    // `with_exact()`). We want to be notified (at compile time) if InstanceMappingPolicy adds,
    // removes, or reorders any members because ProxyInstanceMappingPolicy will have to do the
    // same. By using the aggregate constructor we ensure that such changes will cause compiler
    // errors below.
    return InstanceMappingPolicy{
      p.target.value_or(options.front()), p.allocation, p.ordering, p.exact, p.redundant};
  }();

  std::visit(
    legate::detail::Overload{
      [&](const ProxyInputArguments&) {
        populate_store_mappings(concrete_policy, task.inputs(), store_mappings);
      },
      [&](const ProxyOutputArguments&) {
        populate_store_mappings(concrete_policy, task.outputs(), store_mappings);
      },
      [&](const ProxyReductionArguments&) {
        populate_store_mappings(concrete_policy, task.reductions(), store_mappings);
      },
      [&](const ProxyArrayArgument& arg) {
        const auto kind  = arg.kind;
        const auto index = arg.index;

        switch (kind) {
          case ProxyArrayArgument::Kind::INPUT:
            populate_store_mappings(concrete_policy, {&task.inputs()[index], 1}, store_mappings);
            return;
          case ProxyArrayArgument::Kind::OUTPUT:
            populate_store_mappings(concrete_policy, {&task.outputs()[index], 1}, store_mappings);
            return;
          case ProxyArrayArgument::Kind::REDUCTION:
            populate_store_mappings(
              concrete_policy, {&task.reductions()[index], 1}, store_mappings);
            return;
        }
        LEGATE_ABORT("Unhandled argument kind", legate::detail::to_underlying(kind));
      }},
    stores());
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
