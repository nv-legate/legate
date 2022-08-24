/* Copyright 2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include "core/data/logical_store_detail.h"
#include "core/data/scalar.h"
#include "core/utilities/buffer_builder.h"

namespace legate {

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
  virtual ~ScalarArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override { buffer.pack(value_); }

 private:
  T value_;
};

struct UntypedScalarArg : public ArgWrapper {
 public:
  UntypedScalarArg(const Scalar& scalar) : scalar_(scalar) {}

 public:
  virtual ~UntypedScalarArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

struct RegionFieldArg : public ArgWrapper {
 public:
  RegionFieldArg(RequirementAnalyzer* analyzer,
                 detail::LogicalStore* store,
                 Legion::FieldID field_id,
                 Legion::PrivilegeMode privilege,
                 const ProjectionInfo* proj_info);

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 public:
  virtual ~RegionFieldArg() {}

 private:
  RequirementAnalyzer* analyzer_;
  detail::LogicalStore* store_;
  Legion::LogicalRegion region_;
  Legion::FieldID field_id_;
  Legion::PrivilegeMode privilege_;
  const ProjectionInfo* proj_info_;
};

struct OutputRegionArg : public ArgWrapper {
 public:
  OutputRegionArg(OutputRequirementAnalyzer* analyzer,
                  detail::LogicalStore* store,
                  Legion::FieldSpace field_space,
                  Legion::FieldID field_id);

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 public:
  virtual ~OutputRegionArg() {}

 public:
  detail::LogicalStore* store() const { return store_; }
  const Legion::FieldSpace& field_space() const { return field_space_; }
  Legion::FieldID field_id() const { return field_id_; }
  uint32_t requirement_index() const { return requirement_index_; }

 private:
  OutputRequirementAnalyzer* analyzer_;
  detail::LogicalStore* store_;
  Legion::FieldSpace field_space_;
  Legion::FieldID field_id_;
  mutable uint32_t requirement_index_{-1U};
};

struct FutureStoreArg : public ArgWrapper {
 public:
  FutureStoreArg(detail::LogicalStore* store,
                 bool read_only,
                 bool has_storage,
                 Legion::ReductionOpID redop);

 public:
  virtual ~FutureStoreArg() {}

 public:
  virtual void pack(BufferBuilder& buffer) const override;

 private:
  detail::LogicalStore* store_;
  bool read_only_;
  bool has_storage_;
  Legion::ReductionOpID redop_;
};

}  // namespace legate
