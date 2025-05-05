/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legion.h>

namespace legate::detail {

class Library;

namespace comm::cal {

void register_tasks(detail::Library& core_library);

void register_factory(const detail::Library& core_library);

}  // namespace comm::cal

}  // namespace legate::detail
