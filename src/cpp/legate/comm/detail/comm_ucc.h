/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace legate {

class Library;

}  // namespace legate

namespace legate::detail {

class Library;
class CommunicatorFactory;

}  // namespace legate::detail

namespace legate::detail::comm::ucc {

void register_tasks(const legate::Library& core_library);

[[nodiscard]] std::unique_ptr<CommunicatorFactory> make_factory(const detail::Library& library);

}  // namespace legate::detail::comm::ucc
