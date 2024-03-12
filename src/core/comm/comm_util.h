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

#include "core/runtime/detail/library.h"
#include "core/utilities/typedefs.h"

#include "legion.h"

#include <string_view>

namespace legate::detail {

Legion::TaskVariantRegistrar make_registrar(const detail::Library* library,
                                            std::int64_t local_task_id,
                                            std::string_view task_name,
                                            Processor::Kind proc_kind,
                                            bool concurrent);

}  // namespace legate::detail
