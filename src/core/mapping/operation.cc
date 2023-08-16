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

int64_t Task::task_id() const { return impl_->task_id(); }

namespace {

template <typename Arrays>
std::vector<Array> convert_arrays(const Arrays& arrays)
{
  std::vector<Array> result;
  for (auto& array : arrays) { result.emplace_back(array.get()); }
  return std::move(result);
}

}  // namespace

std::vector<Array> Task::inputs() const { return convert_arrays(impl_->inputs()); }

std::vector<Array> Task::outputs() const { return convert_arrays(impl_->outputs()); }

std::vector<Array> Task::reductions() const { return convert_arrays(impl_->reductions()); }

const std::vector<Scalar>& Task::scalars() const { return impl_->scalars(); }

Array Task::input(uint32_t index) const { return Array(impl_->inputs().at(index).get()); }

Array Task::output(uint32_t index) const { return Array(impl_->outputs().at(index).get()); }

Array Task::reduction(uint32_t index) const { return Array(impl_->reductions().at(index).get()); }

size_t Task::num_inputs() const { return impl_->inputs().size(); }

size_t Task::num_outputs() const { return impl_->outputs().size(); }

size_t Task::num_reductions() const { return impl_->reductions().size(); }

Task::Task(detail::Task* impl) : impl_(impl) {}

// The impl is owned by the caller, so we don't need to deallocate it
Task::~Task() {}

}  // namespace legate::mapping
