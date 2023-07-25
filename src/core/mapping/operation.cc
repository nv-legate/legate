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
#include "core/mapping/detail/operation.h"

namespace legate::mapping {

int64_t Task::task_id() const { return impl_->task_id(); }

namespace {

template <typename Stores>
std::vector<Store> convert_stores(const Stores& stores)
{
  std::vector<Store> result;
  for (auto& store : stores) { result.emplace_back(&store); }
  return std::move(result);
}

}  // namespace

std::vector<Store> Task::inputs() const { return convert_stores(impl_->inputs()); }

std::vector<Store> Task::outputs() const { return convert_stores(impl_->outputs()); }

std::vector<Store> Task::reductions() const { return convert_stores(impl_->reductions()); }

const std::vector<Scalar>& Task::scalars() const { return impl_->scalars(); }

Store Task::input(uint32_t index) const { return Store(&impl_->inputs().at(index)); }

Store Task::output(uint32_t index) const { return Store(&impl_->outputs().at(index)); }

Store Task::reduction(uint32_t index) const { return Store(&impl_->reductions().at(index)); }

size_t Task::num_inputs() const { return impl_->inputs().size(); }

size_t Task::num_outputs() const { return impl_->outputs().size(); }

size_t Task::num_reductions() const { return impl_->reductions().size(); }

Task::Task(detail::Task* impl) : impl_(impl) {}

// The impl is owned by the caller, so we don't need to deallocate it
Task::~Task() {}

}  // namespace legate::mapping
