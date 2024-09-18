/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/data/scalar.h"
#include "legate/mapping/detail/array.h"
#include "legate/mapping/detail/machine.h"
#include "legate/mapping/detail/store.h"
#include "legate/mapping/mapping.h"
#include "legate/utilities/detail/core_ids.h"
#include "legate/utilities/detail/deserializer.h"
#include "legate/utilities/internal_shared_ptr.h"

#include <cstddef>
#include <cstdint>
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
  [[nodiscard]] std::int32_t priority() const;

 protected:
  Mappable() = default;

  mapping::detail::Machine machine_{};
  std::uint32_t sharding_id_{};
  std::int32_t priority_{static_cast<std::int32_t>(legate::detail::TaskPriority::DEFAULT)};

 private:
  struct private_tag {};

  Mappable(private_tag, MapperDataDeserializer dez);
};

class Task : public Mappable {
 public:
  Task(const Legion::Task* task,
       Legion::Mapping::MapperRuntime* runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] LocalTaskID task_id() const;
  [[nodiscard]] legate::detail::Library* library();
  [[nodiscard]] const legate::detail::Library* library() const;

  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& inputs() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& outputs() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<Array>>& reductions() const;
  [[nodiscard]] const std::vector<InternalSharedPtr<legate::detail::Scalar>>& scalars() const;

  [[nodiscard]] const DomainPoint& point() const;

  [[nodiscard]] TaskTarget target() const;

 private:
  const Legion::Task* task_{};
  legate::detail::Library* library_{};

  std::vector<InternalSharedPtr<Array>> inputs_{};
  std::vector<InternalSharedPtr<Array>> outputs_{};
  std::vector<InternalSharedPtr<Array>> reductions_{};
  std::vector<InternalSharedPtr<legate::detail::Scalar>> scalars_{};
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

  [[nodiscard]] const DomainPoint& point() const;

 private:
  const Legion::Copy* copy_{};

  std::vector<Store> inputs_{};
  std::vector<Store> outputs_{};
  std::vector<Store> input_indirections_{};
  std::vector<Store> output_indirections_{};
};

}  // namespace legate::mapping::detail

#include "legate/mapping/detail/operation.inl"
