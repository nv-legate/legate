/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/variant_info.h>

#include <legate/task/variant_info.h>

namespace legate {

const VariantOptions& VariantInfo::options() const noexcept { return impl_().options; }

}  // namespace legate
