/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/scalar.h>
#include <legate/data/scalar.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/internal_shared_ptr.h>

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
  Serializable()                                   = default;
  virtual ~Serializable()                          = default;
  Serializable(const Serializable&)                = default;
  Serializable& operator=(const Serializable&)     = default;
  Serializable(Serializable&&) noexcept            = default;
  Serializable& operator=(Serializable&&) noexcept = default;

  virtual void pack(BufferBuilder& buffer) const = 0;
};

class Analyzable {
 public:
  Analyzable()                                      = default;
  virtual ~Analyzable()                             = default;
  Analyzable(const Analyzable&)                     = default;
  Analyzable& operator=(const Analyzable&) noexcept = default;
  Analyzable(Analyzable&&) noexcept                 = default;
  Analyzable& operator=(Analyzable&&) noexcept      = default;

  virtual void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const = 0;
  virtual void analyze(StoreAnalyzer& analyzer)                                 = 0;

  [[nodiscard]] virtual std::optional<Legion::ProjectionID> get_key_proj_id() const;
  virtual void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const;
  virtual void perform_invalidations() const;
};

class ScalarArg final : public Serializable {
 public:
  explicit ScalarArg(InternalSharedPtr<Scalar> scalar);

  void pack(BufferBuilder& buffer) const override;

 private:
  InternalSharedPtr<Scalar> scalar_{};
};

class RegionFieldArg final : public Analyzable {
 public:
  RegionFieldArg(LogicalStore* store, Legion::PrivilegeMode privilege, StoreProjection store_proj);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void perform_invalidations() const override;

 private:
  LogicalStore* store_{};
  Legion::PrivilegeMode privilege_{};
  StoreProjection store_proj_{};
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

class ScalarStoreArg final : public Analyzable {
 public:
  ScalarStoreArg(LogicalStore* store,
                 Legion::Future future,
                 std::size_t scalar_offset,
                 bool read_only,
                 GlobalRedopID redop);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;

 private:
  LogicalStore* store_{};
  Legion::Future future_{};
  std::size_t scalar_offset_{};
  bool read_only_{};
  GlobalRedopID redop_{};
};

class ReplicatedScalarStoreArg final : public Analyzable {
 public:
  ReplicatedScalarStoreArg(LogicalStore* store,
                           Legion::FutureMap future_map,
                           std::size_t scalar_offset,
                           bool read_only);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;

 private:
  LogicalStore* store_{};
  Legion::FutureMap future_map_{};
  std::size_t scalar_offset_{};
  bool read_only_{};
};

class WriteOnlyScalarStoreArg final : public Analyzable {
 public:
  WriteOnlyScalarStoreArg(LogicalStore* store, GlobalRedopID redop);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;

 private:
  LogicalStore* store_{};
  GlobalRedopID redop_{};
};

class BaseArrayArg final : public Analyzable {
 public:
  BaseArrayArg(std::unique_ptr<Analyzable> data,
               std::optional<std::unique_ptr<Analyzable>> null_mask);

  explicit BaseArrayArg(std::unique_ptr<Analyzable> data);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  std::unique_ptr<Analyzable> data_{};
  std::optional<std::unique_ptr<Analyzable>> null_mask_{};
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
                 std::optional<std::unique_ptr<Analyzable>> null_mask,
                 std::vector<std::unique_ptr<Analyzable>>&& fields);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(std::vector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::optional<std::unique_ptr<Analyzable>> null_mask_{};
  std::vector<std::unique_ptr<Analyzable>> fields_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/launcher_arg.inl>
