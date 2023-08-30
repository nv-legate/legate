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

#include "core/mapping/detail/machine.h"

namespace legate::mapping::detail {
struct Machine;
}  // namespace legate::mapping::detail

namespace legate::detail {

class MachineManager {
 public:
  const mapping::detail::Machine& get_machine() const;

  void push_machine(mapping::detail::Machine&& machine);

  void pop_machine();

 private:
  std::vector<legate::mapping::detail::Machine> machines_;
};

}  // namespace legate::detail
