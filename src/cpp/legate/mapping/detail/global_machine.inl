/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/global_machine.h>

namespace legate::mapping::detail {

inline std::uint32_t GlobalMachine::total_nodes() const { return total_nodes_; }

}  // namespace legate::mapping::detail
