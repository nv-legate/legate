/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail::comm::coll {

void init();

void finalize();

[[noreturn]] void abort() noexcept;

}  // namespace legate::detail::comm::coll
