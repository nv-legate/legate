/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate {

/**
 * @brief Returns the store target that a temporary allocation should default to given the
 * processor currently executing the task.
 *
 * @param host_accessible Whether the allocation must be accessible from the host. Only affects
 * GPU tasks, which pick zero-copy memory when host access is required and framebuffer memory
 * otherwise.
 *
 * @return The store target for the executing processor.
 */
[[nodiscard]] LEGATE_EXPORT mapping::StoreTarget find_store_target_for_executing_processor(
  bool host_accessible = true);

}  // namespace legate
