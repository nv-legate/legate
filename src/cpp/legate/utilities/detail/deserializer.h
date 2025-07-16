/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/communicator.h>
#include <legate/data/detail/physical_array.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/scalar.h>
#include <legate/data/detail/transform.h>
#include <legate/data/physical_store.h>
#include <legate/data/scalar.h>
#include <legate/mapping/detail/array.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/store.h>
#include <legate/type/detail/types.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace legate::detail {

template <typename Deserializer>
class BaseDeserializer {
  BaseDeserializer(const void* args, std::size_t arglen);

  friend Deserializer;

 public:
  template <typename T>
  [[nodiscard]] T unpack();

  template <typename T, std::enable_if_t<type_code_of_v<T> != Type::Code::NIL>* = nullptr>
  void unpack_impl(T& value);

  template <typename T>
  void unpack_impl(std::vector<T>& values);

  template <typename T, std::uint32_t SIZE>
  void unpack_impl(SmallVector<T, SIZE>& values);

  template <typename T1, typename T2>
  void unpack_impl(std::pair<T1, T2>& values);

  [[nodiscard]] std::vector<InternalSharedPtr<detail::Scalar>> unpack_scalars();
  [[nodiscard]] InternalSharedPtr<detail::Scalar> unpack_scalar();
  void unpack_impl(mapping::TaskTarget& value);
  void unpack_impl(mapping::ProcessorRange& value);
  void unpack_impl(mapping::detail::Machine& value);
  void unpack_impl(Domain& domain);

  [[nodiscard]] Span<const std::int8_t> current_args() const;

 protected:
  [[nodiscard]] InternalSharedPtr<TransformStack> unpack_transform_();
  [[nodiscard]] InternalSharedPtr<Type> unpack_type_();

  Span<const std::int8_t> args_{};
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task& task, const std::vector<Legion::PhysicalRegion>& regions);

  using BaseDeserializer::unpack_impl;

  [[nodiscard]] std::vector<InternalSharedPtr<PhysicalArray>> unpack_arrays();
  [[nodiscard]] InternalSharedPtr<PhysicalArray> unpack_array();
  [[nodiscard]] InternalSharedPtr<BasePhysicalArray> unpack_base_array();
  [[nodiscard]] InternalSharedPtr<ListPhysicalArray> unpack_list_array();
  [[nodiscard]] InternalSharedPtr<StructPhysicalArray> unpack_struct_array();
  [[nodiscard]] InternalSharedPtr<PhysicalStore> unpack_store();

  void unpack_impl(FutureWrapper& value);
  void unpack_impl(RegionField& value);
  void unpack_impl(UnboundRegionField& value);
  void unpack_impl(legate::comm::Communicator& value);
  void unpack_impl(Legion::PhaseBarrier& barrier);

 private:
  std::reference_wrapper<const Legion::Task> legion_task_;
  Span<const Legion::Future> futures_{};
  Span<const Legion::PhysicalRegion> regions_{};
  std::vector<Legion::OutputRegion> outputs_{};
};

}  // namespace legate::detail

namespace legate::mapping::detail {

class MapperDataDeserializer : public legate::detail::BaseDeserializer<MapperDataDeserializer> {
 public:
  explicit MapperDataDeserializer(const Legion::Mappable& mappable);

  using BaseDeserializer::unpack_impl;
};

class TaskDeserializer : public legate::detail::BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task& task,
                   Legion::Mapping::MapperRuntime& runtime,
                   Legion::Mapping::MapperContext context);

  using BaseDeserializer::unpack_impl;

  [[nodiscard]] std::vector<InternalSharedPtr<Array>> unpack_arrays();
  [[nodiscard]] InternalSharedPtr<Array> unpack_array();
  [[nodiscard]] InternalSharedPtr<BaseArray> unpack_base_array();
  [[nodiscard]] InternalSharedPtr<ListArray> unpack_list_array();
  [[nodiscard]] InternalSharedPtr<StructArray> unpack_struct_array();
  [[nodiscard]] InternalSharedPtr<Store> unpack_store();

  void unpack_impl(Array& array);
  void unpack_impl(Store& store);
  void unpack_impl(FutureWrapper& value);
  void unpack_impl(RegionField& value, bool unbound);

 private:
  std::reference_wrapper<const Legion::Task> task_;
  std::reference_wrapper<Legion::Mapping::MapperRuntime> runtime_;
  Legion::Mapping::MapperContext context_{};
};

class CopyDeserializer : public legate::detail::BaseDeserializer<CopyDeserializer> {
  using Requirements = std::vector<Legion::RegionRequirement>;
  using ReqsRef      = std::reference_wrapper<const Requirements>;

 public:
  CopyDeserializer(const Legion::Copy& copy,
                   Span<const ReqsRef> all_requirements,
                   Legion::Mapping::MapperRuntime& runtime,
                   Legion::Mapping::MapperContext context);

  using BaseDeserializer::unpack_impl;

  void next_requirement_list();

  void unpack_impl(Store& store);
  void unpack_impl(RegionField& value);

 private:
  Span<const ReqsRef> all_reqs_{};
  const ReqsRef* curr_reqs_{};
  std::reference_wrapper<Legion::Mapping::MapperRuntime> runtime_;
  Legion::Mapping::MapperContext context_{};
  std::uint32_t req_index_offset_{};
};

}  // namespace legate::mapping::detail

#include <legate/utilities/detail/deserializer.inl>
