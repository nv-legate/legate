/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/scalar.h>
#include <legate/mapping/detail/array.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/store.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/streaming.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/deserializer.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping::detail {

class Mappable {
 public:
  explicit Mappable(const Legion::Mappable& mappable);

  /**
   * @brief Deserialize only the streaming generation from the Legion mappable.
   *
   * This is used as an optimization in `BaseMapper::select_tasks_to_map()` to avoid
   * deserializing a bunch of other unrelated mapper information when determining whether to
   * treat a task as a streaming task or not.
   *
   * @param mappable The legion mappable object to deserialize from.
   *
   * @return The streaming generation.
   */
  [[nodiscard]] static std::optional<legate::detail::StreamingGeneration>
  deserialize_only_streaming_generation(const Legion::Mappable& mappable);

  [[nodiscard]] const mapping::detail::Machine& machine() const;
  [[nodiscard]] std::uint32_t sharding_id() const;
  [[nodiscard]] std::int32_t priority() const;

  /**
   * @return The streaming generation.
   */
  [[nodiscard]] const std::optional<legate::detail::StreamingGeneration>& streaming_generation()
    const;

 protected:
  Mappable() = default;

  std::optional<legate::detail::StreamingGeneration> streaming_gen_{};
  mapping::detail::Machine machine_{};
  std::uint32_t sharding_id_{};
  std::int32_t priority_{static_cast<std::int32_t>(legate::detail::TaskPriority::DEFAULT)};

 private:
  struct private_tag {};

  Mappable(private_tag, MapperDataDeserializer dez);
};

class Task : public Mappable {
 public:
  Task(const Legion::Task& task,
       Legion::Mapping::MapperRuntime& runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] LocalTaskID task_id() const;
  [[nodiscard]] legate::detail::Library& library();
  [[nodiscard]] const legate::detail::Library& library() const;

  [[nodiscard]] Span<const InternalSharedPtr<Array>> inputs() const;
  [[nodiscard]] Span<const InternalSharedPtr<Array>> outputs() const;
  [[nodiscard]] Span<const InternalSharedPtr<Array>> reductions() const;
  [[nodiscard]] Span<const InternalSharedPtr<legate::detail::Scalar>> scalars() const;

  [[nodiscard]] bool is_single_task() const;
  [[nodiscard]] const DomainPoint& point() const;
  [[nodiscard]] const Domain& get_launch_domain() const;

  [[nodiscard]] TaskTarget target() const;
  [[nodiscard]] Legion::VariantID legion_task_variant() const;

  // This size doesn't include the upper bound for the returned exception
  [[nodiscard]] std::size_t future_size() const;
  [[nodiscard]] bool can_raise_exception() const;

  [[nodiscard]] const Legion::Task& legion_task() const;

 private:
  std::reference_wrapper<const Legion::Task> task_;
  // This is a pointer (not a reference_wrapper) because it cannot be initialized from
  // initializer lists in the constructor. This is because we cannot create the
  // TaskDeserializer object before first constructing the Mappable base class, and so the
  // usual trick of delegating constructors:
  //
  // Task(Foo, Bar) { ...}
  // Task(Foo f) : Task{f, Bar{}} { ... }
  //
  // Doesn't work. But it is for all intents and purposes a reference wrapper.
  legate::detail::Library* library_{};

  legate::detail::SmallVector<InternalSharedPtr<Array>> inputs_{};
  legate::detail::SmallVector<InternalSharedPtr<Array>> outputs_{};
  legate::detail::SmallVector<InternalSharedPtr<Array>> reductions_{};
  legate::detail::SmallVector<InternalSharedPtr<legate::detail::Scalar>> scalars_{};
  std::size_t future_size_{};
  bool can_raise_exception_{};
};

class Copy : public Mappable {
 public:
  Copy(const Legion::Copy& copy,
       Legion::Mapping::MapperRuntime& runtime,
       Legion::Mapping::MapperContext context);

  [[nodiscard]] Span<const Store> inputs() const;
  [[nodiscard]] Span<const Store> outputs() const;
  [[nodiscard]] Span<const Store> input_indirections() const;
  [[nodiscard]] Span<const Store> output_indirections() const;

  [[nodiscard]] const DomainPoint& point() const;

 private:
  std::reference_wrapper<const Legion::Copy> copy_;

  legate::detail::SmallVector<Store> inputs_{};
  legate::detail::SmallVector<Store> outputs_{};
  legate::detail::SmallVector<Store> input_indirections_{};
  legate::detail::SmallVector<Store> output_indirections_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/operation.inl>
