/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>

namespace legate::detail {

inline const Domain& InlineStorage::domain() const { return domain_; }

inline const legate::ExternalAllocation& InlineStorage::alloc_() const { return allocation_; }

}  // namespace legate::detail
