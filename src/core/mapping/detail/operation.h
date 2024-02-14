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
#include "core/utilities/internal_shared_ptr.h"

#include <vector>

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping::detail {

class Mappable {
 public:
  explicit Mappable(const Legion::Mappable* mappable);

  [[nodiscard]] const mapping::detail::Machine& machine() const;
  [[nodiscard]] std::uint32_t sharding_id() const;

 protected:
  Mappable() = default;

  mapping::detail::Machine machine_{};
  std::uint32_t sharding_id_{};

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

  [[nodiscard]] std::int64_t task_id() const;

  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& inputs() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& outputs() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& reductions() const;
  [[nodiscard]] const std::vector<Scalar>& scalars() const;

  [[nodiscard]] DomainPoint point() const;

  [[nodiscard]] TaskTarget target() const;

 private:
  const legate::detail::Library* library_;
  const Legion::Task* task_;

  std::vector<InternalSharedPtr<Array>> inputs_;
  std::vector<InternalSharedPtr<Array>> outputs_;
  std::vector<InternalSharedPtr<Array>> reductions_;
  std::vector<Scalar> scalars_;
};

class Copy : public Mappable {
 public:
  Copy(const Legion::Copy* copy,
       Legion::Mapping::MapperRuntime* runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] const std::vector<Store>& inputs() const;
  [[nodiscard]] const std::vector<Store>& outputs() const;
  [[nodiscard]] const std::vector<Store>& input_indirections() const;
  [[nodiscard]] const std::vector<Store>& output_indirections() const;

  [[nodiscard]] DomainPoint point() const;

 private:
  const Legion::Copy* copy_;

  std::vector<Store> inputs_;
  std::vector<Store> outputs_;
  std::vector<Store> input_indirections_;
  std::vector<Store> output_indirections_;
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/operation.inl"
