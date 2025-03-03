/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace legate::detail {

class Library;

namespace comm::nccl {

void register_tasks(Library* core_library);

void register_factory(const Library* core_library);

}  // namespace comm::nccl

}  // namespace legate::detail
