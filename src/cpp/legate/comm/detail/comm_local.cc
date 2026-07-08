/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/comm/detail/comm_local.h>

#include <legate_defines.h>

#include <legate/comm/coll.h>
#include <legate/comm/detail/comm_cpu_factory.h>
#include <legate/task/task_config.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/macros.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace legate::detail::comm::local {

std::unique_ptr<CommunicatorFactory> make_factory(const detail::Library&) { return nullptr; }

}  // namespace legate::detail::comm::local
