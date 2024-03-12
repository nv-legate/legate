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

#include "core/comm/comm_util.h"

namespace legate::detail {

Legion::TaskVariantRegistrar make_registrar(const detail::Library* library,
                                            std::int64_t local_task_id,
                                            std::string_view task_name,
                                            Processor::Kind proc_kind,
                                            bool concurrent)
{
  auto global_task_id = library->get_task_id(local_task_id);
  Legion::Runtime::get_runtime()->attach_name(
    global_task_id, task_name.data(), false /*mutable*/, true /*local only*/);

  Legion::TaskVariantRegistrar registrar(global_task_id, task_name.data());

  registrar.add_constraint(Legion::ProcessorConstraint(proc_kind));
  registrar.set_leaf(true);
  registrar.global_registration = false;
  registrar.set_concurrent(concurrent);

  return registrar;
}

}  // namespace legate::detail
