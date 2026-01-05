/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

namespace legate {

[[nodiscard]] LEGATE_EXPORT Memory::Kind find_memory_kind_for_executing_processor(
  bool host_accessible = true);

/**
 * @brief Given a memory kind, retrieve the memory handle for it on the current processor.
 *
 * @param kind The memory kind to find.
 *
 * @return The memory handle.
 *
 * @throw std::out_of_range If the current processor does not support the requested memory
 * kind.
 * @throw std::invalid_argument If the memory kind if invalid.
 */
[[nodiscard]] LEGATE_EXPORT Memory find_memory_from_kind(Memory::Kind kind);

}  // namespace legate
