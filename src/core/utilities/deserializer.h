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

#include <memory>

#include "legion.h"

#include "core/comm/communicator.h"
#include "core/data/detail/array.h"
#include "core/data/detail/scalar.h"
#include "core/data/detail/store.h"
#include "core/data/detail/transform.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/mapping/detail/array.h"
#include "core/mapping/detail/machine.h"
#include "core/mapping/detail/store.h"
#include "core/type/detail/type_info.h"
#include "core/type/type_traits.h"
#include "core/utilities/span.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"

namespace legate {

template <typename Deserializer>
class BaseDeserializer {
 public:
  BaseDeserializer(const void* args, size_t arglen);

 public:
  template <typename T>
  T unpack()
  {
    T value;
    static_cast<Deserializer*>(this)->_unpack(value);
    return value;
  }

 public:
  template <typename T, std::enable_if_t<legate_type_code_of<T> != Type::Code::NIL>* = nullptr>
  void _unpack(T& value)
  {
    value = *reinterpret_cast<const T*>(args_.ptr());
    args_ = args_.subspan(sizeof(T));
  }

 public:
  template <typename T>
  void _unpack(std::vector<T>& values)
  {
    auto size = unpack<uint32_t>();
    values.reserve(size);
    for (uint32_t idx = 0; idx < size; ++idx) values.emplace_back(unpack<T>());
  }
  template <typename T1, typename T2>
  void _unpack(std::pair<T1, T2>& values)
  {
    values.first  = unpack<T1>();
    values.second = unpack<T2>();
  }

 public:
  std::vector<Scalar> unpack_scalars();
  std::unique_ptr<detail::Scalar> unpack_scalar();
  void _unpack(mapping::TaskTarget& value);
  void _unpack(mapping::ProcessorRange& value);
  void _unpack(mapping::detail::Machine& value);
  void _unpack(Domain& domain);

 public:
  Span<const int8_t> current_args() const { return args_; }

 protected:
  std::shared_ptr<detail::TransformStack> unpack_transform();
  std::shared_ptr<detail::Type> unpack_type();

 protected:
  Span<const int8_t> args_;
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions);

 public:
  using BaseDeserializer::_unpack;

 public:
  std::vector<std::shared_ptr<detail::Array>> unpack_arrays();
  std::shared_ptr<detail::Array> unpack_array();
  std::shared_ptr<detail::BaseArray> unpack_base_array();
  std::shared_ptr<detail::ListArray> unpack_list_array();
  std::shared_ptr<detail::StructArray> unpack_struct_array();
  std::shared_ptr<detail::Store> unpack_store();

 public:
  void _unpack(detail::FutureWrapper& value);
  void _unpack(detail::RegionField& value);
  void _unpack(detail::UnboundRegionField& value);
  void _unpack(comm::Communicator& value);
  void _unpack(Legion::PhaseBarrier& barrier);

 private:
  Span<const Legion::Future> futures_;
  Span<const Legion::PhysicalRegion> regions_;
  std::vector<Legion::OutputRegion> outputs_;
};

}  // namespace legate

namespace legate::mapping {

class MapperDataDeserializer : public BaseDeserializer<MapperDataDeserializer> {
 public:
  MapperDataDeserializer(const Legion::Mappable* mappable);

 public:
  using BaseDeserializer::_unpack;
};

class TaskDeserializer : public BaseDeserializer<TaskDeserializer> {
 public:
  TaskDeserializer(const Legion::Task* task,
                   Legion::Mapping::MapperRuntime* runtime,
                   Legion::Mapping::MapperContext context);

 public:
  using BaseDeserializer::_unpack;

 public:
  std::vector<std::shared_ptr<detail::Array>> unpack_arrays();
  std::shared_ptr<detail::Array> unpack_array();
  std::shared_ptr<detail::BaseArray> unpack_base_array();
  std::shared_ptr<detail::ListArray> unpack_list_array();
  std::shared_ptr<detail::StructArray> unpack_struct_array();
  std::shared_ptr<detail::Store> unpack_store();

 public:
  void _unpack(detail::Array& array);
  void _unpack(detail::Store& store);
  void _unpack(detail::FutureWrapper& value);
  void _unpack(detail::RegionField& value, bool is_output_region);

 private:
  const Legion::Task* task_;
  Legion::Mapping::MapperRuntime* runtime_;
  Legion::Mapping::MapperContext context_;
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

 public:
  using BaseDeserializer::_unpack;

 public:
  void next_requirement_list();

 public:
  void _unpack(detail::Store& store);
  void _unpack(detail::RegionField& value);

 private:
  std::vector<ReqsRef> all_reqs_;
  std::vector<ReqsRef>::iterator curr_reqs_;
  Legion::Mapping::MapperRuntime* runtime_;
  Legion::Mapping::MapperContext context_;
  uint32_t req_index_offset_;
};

}  // namespace legate::mapping

#include "core/utilities/deserializer.inl"
