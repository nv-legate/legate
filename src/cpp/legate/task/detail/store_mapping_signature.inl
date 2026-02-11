/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/store_mapping_signature.h>

namespace legate::detail {

inline StoreMappingSignature::StoreMappingSignature(
  SmallVector<InternalSharedPtr<mapping::detail::ProxyStoreMapping>> store_mappings)
  : store_mappings_{std::move(store_mappings)}
{
}

inline Span<const InternalSharedPtr<mapping::detail::ProxyStoreMapping>>
StoreMappingSignature::store_mappings() const
{
  return store_mappings_;
}

}  // namespace legate::detail
