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

#include "core/runtime/detail/machine_manager.h"

#include <stdexcept>

namespace legate::detail {

////////////////////////////////////////////
// legate::detail::MachineManager
////////////////////////////////////////////

const mapping::detail::Machine& MachineManager::get_machine() const noexcept
{
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(!machines_.empty());
  return machines_.top();
}

void MachineManager::push_machine(mapping::detail::Machine&& machine)
{
  machines_.emplace(std::move(machine));
}

void MachineManager::pop_machine()
{
  if (machines_.size() <= 1) throw std::underflow_error{"can't pop from the empty machine stack"};
  machines_.pop();
}

}  // namespace legate::detail
