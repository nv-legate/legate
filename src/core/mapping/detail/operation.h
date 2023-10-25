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
#include "core/utilities/deserializer.h"

#include <memory>
#include <vector>

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping::detail {

namespace {
using Arrays = std::vector<std::shared_ptr<Array>>;
using Stores = std::vector<Store>;
}  // namespace

class Mappable {
 public:
  explicit Mappable(const Legion::Mappable* mappable);

  [[nodiscard]] const mapping::detail::Machine& machine() const;
  [[nodiscard]] uint32_t sharding_id() const;

 protected:
  Mappable() = default;

  mapping::detail::Machine machine_{};
  uint32_t sharding_id_{};

 private:
  struct private_tag {};

  Mappable(private_tag, MapperDataDeserializer dez);
};

class Task : public Mappable {
 public:
  Task(const Legion::Task* task,
       const legate::detail::Library* library,
       Legion::Mapping::MapperRuntime* runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] int64_t task_id() const;

  [[nodiscard]] const Arrays& inputs() const;
  [[nodiscard]] const Arrays& outputs() const;
  [[nodiscard]] const Arrays& reductions() const;
  [[nodiscard]] const std::vector<Scalar>& scalars() const;

  [[nodiscard]] DomainPoint point() const;

  [[nodiscard]] TaskTarget target() const;

 private:
  const legate::detail::Library* library_;
  const Legion::Task* task_;

  Arrays inputs_;
  Arrays outputs_;
  Arrays reductions_;
  std::vector<Scalar> scalars_;
};

class Copy : public Mappable {
 public:
  Copy(const Legion::Copy* copy,
       Legion::Mapping::MapperRuntime* runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] const Stores& inputs() const;
  [[nodiscard]] const Stores& outputs() const;
  [[nodiscard]] const Stores& input_indirections() const;
  [[nodiscard]] const Stores& output_indirections() const;

  [[nodiscard]] DomainPoint point() const;

 private:
  const Legion::Copy* copy_;

  Stores inputs_;
  Stores outputs_;
  Stores input_indirections_;
  Stores output_indirections_;
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/operation.inl"
