/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail {

class Library;

namespace comm {

void register_tasks(Library& library);

void register_builtin_communicator_factories(const Library& library);

}  // namespace comm

}  // namespace legate::detail
