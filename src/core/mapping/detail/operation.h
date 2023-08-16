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

#include "core/data/detail/scalar.h"
#include "core/mapping/detail/array.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/store.h"

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping::detail {

namespace {
using Arrays = std::vector<std::shared_ptr<Array>>;
using Stores = std::vector<Store>;
}  // namespace

class Mappable {
 protected:
  Mappable();

 public:
  Mappable(const Legion::Mappable* mappable);

 public:
  const mapping::detail::Machine& machine() const { return machine_; }
  uint32_t sharding_id() const { return sharding_id_; }

 protected:
  mapping::detail::Machine machine_;
  uint32_t sharding_id_;
};

class Task : public Mappable {
 public:
  Task(const Legion::Task* task,
       const legate::detail::Library* library,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  int64_t task_id() const;

 public:
  const Arrays& inputs() const { return inputs_; }
  const Arrays& outputs() const { return outputs_; }
  const Arrays& reductions() const { return reductions_; }
  const std::vector<Scalar>& scalars() const { return scalars_; }

 public:
  DomainPoint point() const { return task_->index_point; }

 public:
  TaskTarget target() const;

 private:
  const legate::detail::Library* library_;
  const Legion::Task* task_;

 private:
  Arrays inputs_, outputs_, reductions_;
  std::vector<Scalar> scalars_;
};

class Copy : public Mappable {
 public:
  Copy(const Legion::Copy* copy,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  const Stores& inputs() const { return inputs_; }
  const Stores& outputs() const { return outputs_; }
  const Stores& input_indirections() const { return input_indirections_; }
  const Stores& output_indirections() const { return output_indirections_; }

 public:
  DomainPoint point() const { return copy_->index_point; }

 private:
  const Legion::Copy* copy_;

 private:
  Stores inputs_;
  Stores outputs_;
  Stores input_indirections_;
  Stores output_indirections_;
};

}  // namespace legate::mapping::detail
