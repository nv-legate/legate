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
#include "core/data/scalar.h"
#include "core/operation/detail/store_projection.h"
#include "core/utilities/internal_shared_ptr.h"

#include <memory>
#include <optional>
#include <vector>

namespace legate::detail {

class BufferBuilder;
class OutputRegionArg;
class LogicalStore;
class StoreAnalyzer;

class Serializable {
 public:
  virtual ~Serializable() = default;

  virtual void pack(BufferBuilder& buffer) const = 0;
};

class Analyzable {
 public:
  virtual ~Analyzable() = default;

  virtual void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const = 0;
  virtual void analyze(StoreAnalyzer& analyzer)                                 = 0;

  [[nodiscard]] virtual std::optional<Legion::ProjectionID> get_key_proj_id() const;
  virtual void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const;
  virtual void perform_invalidations() const;
};

class ScalarArg final : public Serializable {
 public:
  explicit ScalarArg(Scalar&& scalar);

  void pack(BufferBuilder& buffer) const override;

 private:
  Scalar scalar_;
};

class RegionFieldArg final : public Analyzable {
 public:
  RegionFieldArg(LogicalStore* store,
                 Legion::PrivilegeMode privilege,
                 std::unique_ptr<StoreProjection> store_proj);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void perform_invalidations() const override;

 private:
  LogicalStore* store_{};
  Legion::PrivilegeMode privilege_{};
  std::unique_ptr<StoreProjection> store_proj_{};
};

class OutputRegionArg final : public Analyzable {
 public:
  OutputRegionArg(LogicalStore* store, Legion::FieldSpace field_space, Legion::FieldID field_id);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;

  [[nodiscard]] LogicalStore* store() const;
  [[nodiscard]] const Legion::FieldSpace& field_space() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] std::uint32_t requirement_index() const;

 private:
  LogicalStore* store_{};
  Legion::FieldSpace field_space_{};
  Legion::FieldID field_id_{};
  mutable std::uint32_t requirement_index_{-1U};
};

class FutureStoreArg final : public Analyzable {
 public:
  FutureStoreArg(LogicalStore* store,
                 bool read_only,
                 bool has_storage,
                 Legion::ReductionOpID redop);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;

 private:
  LogicalStore* store_{};
  bool read_only_{};
  bool has_storage_{};
  Legion::ReductionOpID redop_{};
};

class BaseArrayArg final : public Analyzable {
 public:
  BaseArrayArg(std::unique_ptr<Analyzable> data, std::unique_ptr<Analyzable> null_mask);

  explicit BaseArrayArg(std::unique_ptr<Analyzable> data);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  std::unique_ptr<Analyzable> data_{};
  std::unique_ptr<Analyzable> null_mask_{};
};

class ListArrayArg final : public Analyzable {
 public:
  ListArrayArg(InternalSharedPtr<Type> type,
               std::unique_ptr<Analyzable> descriptor,
               std::unique_ptr<Analyzable> vardata);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::unique_ptr<Analyzable> descriptor_{};
  std::unique_ptr<Analyzable> vardata_{};
};

class StructArrayArg final : public Analyzable {
 public:
  StructArrayArg(InternalSharedPtr<Type> type,
                 std::unique_ptr<Analyzable> null_mask,
                 std::vector<std::unique_ptr<Analyzable>>&& fields);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::unique_ptr<Analyzable> null_mask_{};
  std::vector<std::unique_ptr<Analyzable>> fields_{};
};

}  // namespace legate::detail

#include "core/operation/detail/launcher_arg.inl"
