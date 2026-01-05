/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/operation.h>

#include <legate/mapping/detail/array.h>
#include <legate/mapping/detail/operation.h>

namespace legate::mapping {

LocalTaskID Task::task_id() const { return impl()->task_id(); }

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

std::vector<Scalar> Task::scalars() const
{
  auto&& scals = impl()->scalars();

  return {scals.begin(), scals.end()};
}

Array Task::input(std::uint32_t index) const { return Array{impl()->inputs().at(index).get()}; }

Array Task::output(std::uint32_t index) const { return Array{impl()->outputs().at(index).get()}; }

Array Task::reduction(std::uint32_t index) const
{
  return Array{impl()->reductions().at(index).get()};
}

Scalar Task::scalar(std::uint32_t index) const { return Scalar{impl()->scalars().at(index)}; }

std::size_t Task::num_inputs() const { return impl()->inputs().size(); }

std::size_t Task::num_outputs() const { return impl()->outputs().size(); }

std::size_t Task::num_reductions() const { return impl()->reductions().size(); }

std::size_t Task::num_scalars() const { return impl()->scalars().size(); }

bool Task::is_single_task() const { return impl()->is_single_task(); }

const Domain& Task::get_launch_domain() const { return impl()->get_launch_domain(); }

}  // namespace legate::mapping
