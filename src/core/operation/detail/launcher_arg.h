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

#include "core/data/detail/logical_store.h"
#include "core/data/detail/scalar.h"
#include "core/data/scalar.h"

namespace legate::detail {

class BufferBuilder;
class OutputRequirementAnalyzer;
class ProjectionInfo;
class RequirementAnalyzer;

struct ArgWrapper {
  virtual ~ArgWrapper() {}
  virtual void pack(BufferBuilder& buffer) const = 0;
};

template <typename T>
struct ScalarArg : public ArgWrapper {
 public:
  ScalarArg(const T& value) : value_(value) {}

 public:
  ~ScalarArg() {}

 public:
  void pack(BufferBuilder& buffer) const override { buffer.pack(value_); }

 private:
  T value_;
};

struct UntypedScalarArg : public ArgWrapper {
 public:
  UntypedScalarArg(Scalar&& scalar) : scalar_(std::move(scalar)) {}

 public:
  ~UntypedScalarArg() {}

 public:
  void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

struct RegionFieldArg : public ArgWrapper {
 public:
  RegionFieldArg(RequirementAnalyzer* analyzer,
                 LogicalStore* store,
                 Legion::FieldID field_id,
                 Legion::PrivilegeMode privilege,
                 std::unique_ptr<ProjectionInfo> proj_info);

 public:
  void pack(BufferBuilder& buffer) const override;

 public:
  ~RegionFieldArg() {}

 private:
  RequirementAnalyzer* analyzer_;
  LogicalStore* store_;
  Legion::LogicalRegion region_;
  Legion::FieldID field_id_;
  Legion::PrivilegeMode privilege_;
  std::unique_ptr<ProjectionInfo> proj_info_;
};

struct OutputRegionArg : public ArgWrapper {
 public:
  OutputRegionArg(OutputRequirementAnalyzer* analyzer,
                  LogicalStore* store,
                  Legion::FieldSpace field_space,
                  Legion::FieldID field_id);

 public:
  void pack(BufferBuilder& buffer) const override;

 public:
  ~OutputRegionArg() {}

 public:
  LogicalStore* store() const { return store_; }
  const Legion::FieldSpace& field_space() const { return field_space_; }
  Legion::FieldID field_id() const { return field_id_; }
  uint32_t requirement_index() const { return requirement_index_; }

 private:
  OutputRequirementAnalyzer* analyzer_;
  LogicalStore* store_;
  Legion::FieldSpace field_space_;
  Legion::FieldID field_id_;
  mutable uint32_t requirement_index_{-1U};
};

struct FutureStoreArg : public ArgWrapper {
 public:
  FutureStoreArg(LogicalStore* store,
                 bool read_only,
                 bool has_storage,
                 Legion::ReductionOpID redop);

 public:
  ~FutureStoreArg() {}

 public:
  void pack(BufferBuilder& buffer) const override;

 private:
  LogicalStore* store_;
  bool read_only_;
  bool has_storage_;
  Legion::ReductionOpID redop_;
};

}  // namespace legate::detail
