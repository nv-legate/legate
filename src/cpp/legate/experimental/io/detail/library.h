/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate {

class Library;

}  // namespace legate

namespace legate::experimental::io::detail {

[[nodiscard]] legate::Library& core_io_library();

}  // namespace legate::experimental::io::detail
