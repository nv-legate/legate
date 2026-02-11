/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/proxy_store_mapping.h>

namespace legate::mapping {

inline const SharedPtr<detail::ProxyStoreMapping>& ProxyStoreMapping::impl() const { return impl_; }

}  // namespace legate::mapping
