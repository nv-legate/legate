/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/typedefs.h>

namespace legate {

[[nodiscard]] Memory::Kind find_memory_kind_for_executing_processor(bool host_accessible = true);

}  // namespace legate
