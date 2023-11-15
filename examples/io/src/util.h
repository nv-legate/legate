/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/store.h"
#include "core/task/task_context.h"

#include <filesystem>

namespace legateio {

std::filesystem::path get_unique_path_for_task_index(legate::TaskContext task_context,
                                                     int32_t ndim,
                                                     const std::string& dirname);

void write_to_file(legate::TaskContext task_context,
                   const std::string& dirname,
                   const legate::Store& store);

}  // namespace legateio
