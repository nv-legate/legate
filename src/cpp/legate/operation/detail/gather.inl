/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/gather.h>

namespace legate::detail {

inline void Gather::set_indirect_out_of_range(bool flag) { out_of_range_ = flag; }

inline Operation::Kind Gather::kind() const { return Kind::GATHER; }

}  // namespace legate::detail
