/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/proxy_store_mapping.h>

namespace legate::mapping::detail {

inline const std::
  variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>&
  ProxyStoreMapping::stores() const
{
  return stores_;
}

inline const ProxyInstanceMappingPolicy& ProxyStoreMapping::policy() const { return policy_; }

}  // namespace legate::mapping::detail
