/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/machine.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <vector>

namespace legate::mapping::detail {

/**
 * @brief Representation of the global machine that is visible to Legate at program startup.
 */
class GlobalMachine {
 public:
  /**
   * @brief Constructs a GlobalMachine object by grouping all procs of a specific type into a
   * vector.
   *
   * @return A GlobalMachine object representing the global machine.
   */
  GlobalMachine();

  /**
   * @brief Returns a vector of processors global machine that match target, in no particular order.
   *
   * @param target The target type of the processors to get.
   *
   * @return A vector of processors on the global machine matching the task target.
   */
  [[nodiscard]] legate::Span<const Processor> procs(TaskTarget target) const;

  /**
   * @brief Returns a span of processors from the global machine that are within the scope of the
   * machine and its preferred target.
   *
   * @param machine The machine scope to also slice processors from.
   *
   * @return A span of processors respecting the target type on global machine and within machine
   * slice, if possible. Otherwise, returns an empty span.
   */
  [[nodiscard]] ProcessorSpan slice(const Machine& machine) const;

  /**
   * @brief Returns a span of processors from the global machine that are within the scope of the
   * machine and its preferred target. This method ignores machine if machine does not intersect
   * with the object.
   *
   * @param machine The machine scope to slice processors from its preferred target.
   *
   * @return A span of processors respecting the target type on global machine and within machine
   * slice, if possible. Otherwise, returns all processors on this machine for the preferred target.
   */
  [[nodiscard]] ProcessorSpan slice_with_fallback(const Machine& machine) const;

  /** @brief Returns the total number of nodes (i.e. number of ranks) on the system. */
  [[nodiscard]] std::uint32_t total_nodes() const;

 private:
  std::uint32_t total_nodes_{};

  // global lists of processors viewable by legate
  std::vector<Processor> global_cpus_{};
  std::vector<Processor> global_gpus_{};
  std::vector<Processor> global_omps_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/global_machine.inl>
