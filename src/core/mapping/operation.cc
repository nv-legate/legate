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

#include "core/mapping/operation.h"

#include "core/mapping/detail/array.h"
#include "core/mapping/detail/operation.h"

namespace legate::mapping {

std::int64_t Task::task_id() const { return impl()->task_id(); }

namespace {

template <typename Arrays>
std::vector<Array> convert_arrays(const Arrays& arrays)
{
  std::vector<Array> result;

  result.reserve(arrays.size());
  for (auto&& array : arrays) {
    result.emplace_back(array.get());
  }
  return result;
}

}  // namespace

std::vector<Array> Task::inputs() const { return convert_arrays(impl()->inputs()); }

std::vector<Array> Task::outputs() const { return convert_arrays(impl()->outputs()); }

std::vector<Array> Task::reductions() const { return convert_arrays(impl()->reductions()); }

const std::vector<Scalar>& Task::scalars() const { return impl()->scalars(); }

Array Task::input(std::uint32_t index) const { return Array{impl()->inputs().at(index).get()}; }

Array Task::output(std::uint32_t index) const { return Array{impl()->outputs().at(index).get()}; }

Array Task::reduction(std::uint32_t index) const
{
  return Array{impl()->reductions().at(index).get()};
}

std::size_t Task::num_inputs() const { return impl()->inputs().size(); }

std::size_t Task::num_outputs() const { return impl()->outputs().size(); }

std::size_t Task::num_reductions() const { return impl()->reductions().size(); }

}  // namespace legate::mapping
