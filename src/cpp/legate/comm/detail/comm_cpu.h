/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail {

class Library;

namespace comm::cpu {

void register_tasks(Library& library);

void register_factory(const Library& library);

}  // namespace comm::cpu

}  // namespace legate::detail
