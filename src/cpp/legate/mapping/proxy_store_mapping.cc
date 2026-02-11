/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/proxy_store_mapping.h>

#include <legate/mapping/detail/proxy_store_mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/mapping/proxy_instance_mapping_policy.h>

namespace legate::mapping {

ProxyStoreMapping::ProxyStoreMapping(InternalSharedPtr<detail::ProxyStoreMapping> impl)
  : impl_{std::move(impl)}
{
}

ProxyStoreMapping::ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                                  ProxyInputArguments,
                                                  ProxyOutputArguments,
                                                  ProxyReductionArguments> store,
                                     std::optional<StoreTarget> target,
                                     bool exact)
  : ProxyStoreMapping{std::move(store), [&] {
                        auto ret = ProxyInstanceMappingPolicy{};

                        ret.target = target;
                        ret.exact  = exact;
                        return ret;
                      }()}
{
}

ProxyStoreMapping::ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                                  ProxyInputArguments,
                                                  ProxyOutputArguments,
                                                  ProxyReductionArguments> store,
                                     ProxyInstanceMappingPolicy&& policy)
  : impl_{legate::make_shared<detail::ProxyStoreMapping>(std::move(store), std::move(policy))}
{
}

const ProxyInstanceMappingPolicy& ProxyStoreMapping::policy() const { return impl()->policy(); }

ProxyStoreMapping::~ProxyStoreMapping() = default;

bool operator==(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs)
{
  return *lhs.impl() == *rhs.impl();
}

bool operator!=(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs)
{
  return !(lhs == rhs);
}

}  // namespace legate::mapping
