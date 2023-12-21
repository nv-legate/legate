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

#include "core/comm/communicator.h"
#include "core/data/detail/physical_array.h"
#include "core/data/detail/physical_store.h"
#include "core/data/detail/scalar.h"
#include "core/data/detail/transform.h"
#include "core/data/physical_store.h"
#include "core/data/scalar.h"
#include "core/mapping/detail/array.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/store.h"
#include "core/type/detail/type_info.h"
#include "core/type/type_traits.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"

#include <memory>
#include <utility>
#include <vector>

namespace legate::detail {

template <typename T>
std::pair<void*, std::size_t> align_for_unpack(void* ptr,
                                               std::size_t capacity,
                                               std::size_t bytes = sizeof(T),
                                               std::size_t align = alignof(T));
}  // namespace legate::detail

namespace legate {

template <typename Deserializer>
class BaseDeserializer {
 public:
  BaseDeserializer(const void* args, size_t arglen);

  template <typename T>
  [[nodiscard]] T unpack();

  template <typename T, std::enable_if_t<type_code_of<T> != Type::Code::NIL>* = nullptr>
  void _unpack(T& value);

  template <typename T>
  void _unpack(std::vector<T>& values);

  template <typename T1, typename T2>
  void _unpack(std::pair<T1, T2>& values);

  [[nodiscard]] std::vector<Scalar> unpack_scalars();
  [[nodiscard]] std::unique_ptr<detail::Scalar> unpack_scalar();
  void _unpack(mapping::TaskTarget& value);
  void _unpack(mapping::ProcessorRange& value);
  void _unpack(mapping::detail::Machine& value);
  void _unpack(Domain& domain);

  [[nodiscard]] Span<const int8_t> current_args() const;

 protected:
  [[nodiscard]] InternalSharedPtr<detail::TransformStack> unpack_transform();
  [[nodiscard]] InternalSharedPtr<detail::Type> unpack_type();

  Span<const int8_t> args_{};
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

  using BaseDeserializer::_unpack;

  [[nodiscard]] std::vector<InternalSharedPtr<detail::PhysicalArray>> unpack_arrays();
  [[nodiscard]] InternalSharedPtr<detail::PhysicalArray> unpack_array();
  [[nodiscard]] InternalSharedPtr<detail::BasePhysicalArray> unpack_base_array();
  [[nodiscard]] InternalSharedPtr<detail::ListPhysicalArray> unpack_list_array();
  [[nodiscard]] InternalSharedPtr<detail::StructPhysicalArray> unpack_struct_array();
  [[nodiscard]] InternalSharedPtr<detail::PhysicalStore> unpack_store();

  void _unpack(detail::FutureWrapper& value);
  void _unpack(detail::RegionField& value);
  void _unpack(detail::UnboundRegionField& value);
  void _unpack(comm::Communicator& value);
  void _unpack(Legion::PhaseBarrier& barrier);

 private:
  Span<const Legion::Future> futures_{};
  Span<const Legion::PhysicalRegion> regions_{};
  std::vector<Legion::OutputRegion> outputs_{};
};

}  // namespace legate

namespace legate::mapping {

class MapperDataDeserializer : public BaseDeserializer<MapperDataDeserializer> {
 public:
  explicit MapperDataDeserializer(const Legion::Mappable* mappable);

  using BaseDeserializer::_unpack;
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task,
                   Legion::Mapping::MapperRuntime* runtime,
                   Legion::Mapping::MapperContext context);

  using BaseDeserializer::_unpack;

  [[nodiscard]] std::vector<InternalSharedPtr<detail::Array>> unpack_arrays();
  [[nodiscard]] InternalSharedPtr<detail::Array> unpack_array();
  [[nodiscard]] InternalSharedPtr<detail::BaseArray> unpack_base_array();
  [[nodiscard]] InternalSharedPtr<detail::ListArray> unpack_list_array();
  [[nodiscard]] InternalSharedPtr<detail::StructArray> unpack_struct_array();
  [[nodiscard]] InternalSharedPtr<detail::Store> unpack_store();

  void _unpack(detail::Array& array);
  void _unpack(detail::Store& store);
  void _unpack(detail::FutureWrapper& value);
  void _unpack(detail::RegionField& value, bool is_output_region);

 private:
  const Legion::Task* task_{};
  Legion::Mapping::MapperRuntime* runtime_{};
  Legion::Mapping::MapperContext context_{};
};

class CopyDeserializer : public BaseDeserializer<CopyDeserializer> {
 private:
  using Requirements = std::vector<Legion::RegionRequirement>;
  using ReqsRef      = std::reference_wrapper<const Requirements>;

 public:
  CopyDeserializer(const Legion::Copy* copy,
                   std::vector<ReqsRef>&& all_requirements,
                   Legion::Mapping::MapperRuntime* runtime,
                   Legion::Mapping::MapperContext context);

  using BaseDeserializer::_unpack;

  void next_requirement_list();

  void _unpack(detail::Store& store);
  void _unpack(detail::RegionField& value);

 private:
  std::vector<ReqsRef> all_reqs_{};
  std::vector<ReqsRef>::iterator curr_reqs_{};
  Legion::Mapping::MapperRuntime* runtime_{};
  Legion::Mapping::MapperContext context_{};
  uint32_t req_index_offset_{};
};

}  // namespace legate::mapping

#include "core/utilities/deserializer.inl"
