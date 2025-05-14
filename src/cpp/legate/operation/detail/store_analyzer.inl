/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/store_analyzer.h>

namespace legate::detail {

inline bool RequirementAnalyzer::empty() const { return field_sets_.empty(); }

inline void RequirementAnalyzer::relax_interference_checks(bool relax)
{
  relax_interference_checks_ = relax;
}

// ==========================================================================================

inline bool OutputRequirementAnalyzer::empty() const { return field_groups_.empty(); }

// ==========================================================================================

template <typename Launcher>
inline void StoreAnalyzer::populate(Launcher& launcher,
                                    std::vector<Legion::OutputRequirement>& out_reqs) const
{
  req_analyzer_.populate_launcher(launcher);
  out_analyzer_.populate_output_requirements(out_reqs);
  fut_analyzer_.populate_launcher(launcher);
}

inline bool StoreAnalyzer::can_be_local_function_task() const
{
  return req_analyzer_.empty() && out_analyzer_.empty();
}

inline void StoreAnalyzer::relax_interference_checks(bool relax)
{
  req_analyzer_.relax_interference_checks(relax);
}

}  // namespace legate::detail
