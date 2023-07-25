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

#include "core/data/scalar.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/store.h"

/**
 * @file
 * @brief Class definitions for operations and stores used in mapping
 */

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping::detail {

namespace {
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

/**
 * @ingroup mapping
 * @brief A metadata class for tasks
 */
class Task : public Mappable {
 public:
  Task(const Legion::Task* task,
       const legate::detail::Library* library,
       Legion::Mapping::MapperRuntime* runtime,
       const Legion::Mapping::MapperContext context);

 public:
  /**
   * @brief Returns the task id
   *
   * @return Task id
   */
  int64_t task_id() const;

 public:
  /**
   * @brief Returns metadata for the task's input stores
   *
   * @return Vector of store metadata objects
   */
  const Stores& inputs() const { return inputs_; }
  /**
   * @brief Returns metadata for the task's output stores
   *
   * @return Vector of store metadata objects
   */
  const Stores& outputs() const { return outputs_; }
  /**
   * @brief Returns metadata for the task's reduction stores
   *
   * @return Vector of store metadata objects
   */
  const Stores& reductions() const { return reductions_; }
  /**
   * @brief Returns the vector of the task's by-value arguments. Unlike `mapping::Store`
   * objects that have no access to data in the stores, the returned `Scalar` objects
   * contain valid arguments to the task
   *
   * @return Vector of `Scalar` objects
   */
  const std::vector<legate::Scalar>& scalars() const { return scalars_; }

 public:
  /**
   * @brief Returns the point of the task
   *
   * @return The point of the task
   */
  DomainPoint point() const { return task_->index_point; }

 public:
  TaskTarget target() const;

 private:
  const legate::detail::Library* library_;
  const Legion::Task* task_;

 private:
  Stores inputs_, outputs_, reductions_;
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
