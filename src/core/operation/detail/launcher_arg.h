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

#include <optional>

#include "core/data/detail/logical_store.h"
#include "core/data/detail/scalar.h"
#include "core/data/scalar.h"
#include "core/operation/detail/projection.h"

namespace legate::detail {

class BufferBuilder;
struct OutputRegionArg;
struct StoreAnalyzer;
class TaskLauncher;

struct Serializable {
  virtual ~Serializable() {}
  virtual void pack(BufferBuilder& buffer) const = 0;
};

struct Analyzable {
  virtual ~Analyzable() {}
  virtual void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const = 0;
  virtual void analyze(StoreAnalyzer& analyzer)                                 = 0;
  virtual std::optional<Legion::ProjectionID> get_key_proj_id() const { return std::nullopt; }
  virtual void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const {}
  virtual void perform_invalidations() const {}
};

struct ScalarArg : public Serializable {
 public:
  ScalarArg(Scalar&& scalar) : scalar_(std::move(scalar)) {}

 public:
  ~ScalarArg() {}

 public:
  void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

struct RegionFieldArg : public Analyzable {
 public:
  RegionFieldArg(LogicalStore* store,
                 Legion::PrivilegeMode privilege,
                 std::unique_ptr<ProjectionInfo> proj_info);

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void perform_invalidations() const override;

 public:
  ~RegionFieldArg() {}

 private:
  LogicalStore* store_;
  Legion::PrivilegeMode privilege_;
  std::unique_ptr<ProjectionInfo> proj_info_;
};

struct OutputRegionArg : public Analyzable {
 public:
  OutputRegionArg(LogicalStore* store, Legion::FieldSpace field_space);

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;

 public:
  ~OutputRegionArg() {}

 public:
  LogicalStore* store() const { return store_; }
  const Legion::FieldSpace& field_space() const { return field_space_; }
  Legion::FieldID field_id() const { return field_id_; }
  uint32_t requirement_index() const { return requirement_index_; }

 private:
  LogicalStore* store_;
  Legion::FieldSpace field_space_;
  Legion::FieldID field_id_;
  mutable uint32_t requirement_index_{-1U};
};

struct FutureStoreArg : public Analyzable {
 public:
  FutureStoreArg(LogicalStore* store,
                 bool read_only,
                 bool has_storage,
                 Legion::ReductionOpID redop);

 public:
  ~FutureStoreArg() {}

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;

 private:
  LogicalStore* store_;
  bool read_only_;
  bool has_storage_;
  Legion::ReductionOpID redop_;
};

struct BaseArrayArg : public Analyzable {
 public:
  BaseArrayArg(std::unique_ptr<Analyzable> data);
  BaseArrayArg(std::unique_ptr<Analyzable> data, std::unique_ptr<Analyzable> null_mask);

 public:
  ~BaseArrayArg() {}

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  std::unique_ptr<Analyzable> data_;
  std::unique_ptr<Analyzable> null_mask_{nullptr};
};

struct ListArrayArg : public Analyzable {
 public:
  ListArrayArg(std::shared_ptr<Type> type,
               std::unique_ptr<Analyzable> descriptor,
               std::unique_ptr<Analyzable> vardata);

 public:
  ~ListArrayArg() {}

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  std::shared_ptr<Type> type_;
  std::unique_ptr<Analyzable> descriptor_;
  std::unique_ptr<Analyzable> vardata_;
};

struct StructArrayArg : public Analyzable {
 public:
  StructArrayArg(std::shared_ptr<Type> type,
                 std::unique_ptr<Analyzable> null_mask,
                 std::vector<std::unique_ptr<Analyzable>>&& fields);

 public:
  ~StructArrayArg() {}

 public:
  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  std::shared_ptr<Type> type_;
  std::unique_ptr<Analyzable> null_mask_;
  std::vector<std::unique_ptr<Analyzable>> fields_;
};

}  // namespace legate::detail
