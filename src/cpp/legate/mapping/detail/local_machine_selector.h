/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/utilities/typedefs.h>

#include <unordered_map>

namespace legate::mapping::detail {

/**
 * @brief A cache for retrieving LocalMachine objects without constant re-construction.
 */
class LocalMachineSelector {
 public:
  LocalMachineSelector() = default;

  /**
   * @brief Returns a LocalMachine corresponding to the caller's rank.
   *
   * @return The LocalMachine belonging to the same rank as the caller.
   */
  [[nodiscard]] const LocalMachine& get_local() const;

  /**
   * @brief Returns the LocalMachine that contains Processor p.
   *
   * A LocalMachine "contains" a processor if and only if the Realm address space of the processor
   * is the same as the Realm address space that LocalMachine represents. A LocalMachine in this
   * case will contain all of the processors and memories that are within the same Realm address
   * space.
   *
   * @param p The processor whose rank (i.e. address space) will be used to get the LocalMachine.
   *
   * @return The LocalMachine containing all processors and memories within the same rank as p.
   */
  [[nodiscard]] const LocalMachine& get_local_to(Processor p) const;

  /**
   * @brief Returns the LocalMachine that contains Memory m.
   *
   * A LocalMachine "contains" a memory if and only if the Realm address space of the memory
   * is the same as the Realm address space that LocalMachine represents. A LocalMachine in this
   * case will contain all of the processors and memories that are within the same Realm address
   * space.
   *
   * @param m The memory whose rank (i.e. address space) will be used to get the LocalMachine.
   *
   * @return The LocalMachine containing all processors and memories within the same rank as m.
   */
  [[nodiscard]] const LocalMachine& get_local_to(Memory m) const;

 private:
  mutable std::unordered_map<Realm::AddressSpace, LocalMachine> local_machine_cache_{};
};

}  // namespace legate::mapping::detail
