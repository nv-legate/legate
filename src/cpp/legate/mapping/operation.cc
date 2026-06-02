/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/operation.h>

#include <legate/mapping/detail/operation.h>
#include <legate/mapping/detail/store.h>

namespace legate::mapping {

LocalTaskID Task::task_id() const { return impl()->task_id(); }

namespace {

template <typename Stores>
std::vector<Store> convert_stores(const Stores& stores)
{
  std::vector<Store> result;

  result.reserve(stores.size());
  for (auto&& store : stores) {
    result.emplace_back(store.get());
  }
  return result;
}

}  // namespace

std::vector<Store> Task::inputs() const { return convert_stores(impl()->inputs()); }

std::vector<Store> Task::outputs() const { return convert_stores(impl()->outputs()); }

std::vector<Store> Task::reductions() const { return convert_stores(impl()->reductions()); }

std::vector<Scalar> Task::scalars() const
{
  auto&& scals = impl()->scalars();

  return {scals.begin(), scals.end()};
}

Store Task::input(std::uint32_t index) const { return Store{impl()->inputs().at(index).get()}; }

Store Task::output(std::uint32_t index) const { return Store{impl()->outputs().at(index).get()}; }

Store Task::reduction(std::uint32_t index) const
{
  return Store{impl()->reductions().at(index).get()};
}

Scalar Task::scalar(std::uint32_t index) const { return Scalar{impl()->scalars().at(index)}; }

std::size_t Task::num_inputs() const { return impl()->inputs().size(); }

std::size_t Task::num_outputs() const { return impl()->outputs().size(); }

std::size_t Task::num_reductions() const { return impl()->reductions().size(); }

std::size_t Task::num_scalars() const { return impl()->scalars().size(); }

bool Task::is_single_task() const { return impl()->is_single_task(); }

const Domain& Task::get_launch_domain() const { return impl()->get_launch_domain(); }

}  // namespace legate::mapping
