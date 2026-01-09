/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/scalar.h>
#include <legate/data/scalar.h>
#include <legate/operation/detail/store_projection.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace legate::detail {

class BufferBuilder;
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

class ScalarArg final : public Serializable {
 public:
  explicit ScalarArg(InternalSharedPtr<Scalar> scalar);

  void pack(BufferBuilder& buffer) const override;

 private:
  InternalSharedPtr<Scalar> scalar_{};
};

class RegionFieldArg;
class OutputRegionArg;
class ScalarStoreArg;
class ReplicatedScalarStoreArg;
class WriteOnlyScalarStoreArg;
class BaseArrayArg;
class ListArrayArg;
class StructArrayArg;

using StoreAnalyzable = std::variant<RegionFieldArg,
                                     OutputRegionArg,
                                     ScalarStoreArg,
                                     ReplicatedScalarStoreArg,
                                     WriteOnlyScalarStoreArg>;

using ArrayAnalyzable = std::variant<BaseArrayArg, ListArrayArg, StructArrayArg>;

namespace variant_detail {

template <typename...>
struct variant_concat;

template <typename... T1, typename... T2>
struct variant_concat<std::variant<T1...>, std::variant<T2...>> {
  using type = std::variant<T1..., T2...>;
};

template <typename... T>
using variant_concat_t = typename variant_concat<T...>::type;

template <typename... T>
struct VariantProxy {
  std::variant<T...> v;

  template <typename... ToT>
  constexpr operator std::variant<ToT...>() &&  // NOLINT(google-explicit-constructor)
  {
    return std::visit(
      [](auto&& arg) -> std::variant<ToT...> {
        using arg_type = std::decay_t<decltype(arg)>;

        // This ensures we do not slice. We need to construct the most derived type (in this
        // case, the exact type we came in with). Otherwise, if there is no match, the variant
        // might choose to construct the base class if it is publicly accessible.
        return std::variant<ToT...>{std::in_place_type<arg_type>, std::forward<arg_type>(arg)};
      },
      std::move(v));
  }
};

}  // namespace variant_detail

template <typename... T>
constexpr variant_detail::VariantProxy<T...> variant_cast(std::variant<T...> v)
{
  return {std::move(v)};
}

using Analyzable = variant_detail::variant_concat_t<StoreAnalyzable, ArrayAnalyzable>;

// ==========================================================================================

class AnalyzableBase {
 public:
  AnalyzableBase()                                          = default;
  virtual ~AnalyzableBase()                                 = default;
  AnalyzableBase(const AnalyzableBase&)                     = default;
  AnalyzableBase& operator=(const AnalyzableBase&) noexcept = default;
  AnalyzableBase(AnalyzableBase&&) noexcept                 = default;
  AnalyzableBase& operator=(AnalyzableBase&&) noexcept      = default;

  virtual void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const = 0;
  virtual void analyze(StoreAnalyzer& analyzer) const                           = 0;

  [[nodiscard]] virtual std::optional<Legion::ProjectionID> get_key_proj_id() const;
  virtual void record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const;
  virtual void perform_invalidations() const;
};

// ==========================================================================================

class RegionFieldArg final : public AnalyzableBase {
 public:
  RegionFieldArg(LogicalStore* store, Legion::PrivilegeMode privilege, StoreProjection store_proj);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void perform_invalidations() const override;

 private:
  LogicalStore* store_{};
  Legion::PrivilegeMode privilege_{};
  StoreProjection store_proj_{};
};

class OutputRegionArg final : public AnalyzableBase {
 public:
  OutputRegionArg(LogicalStore* store, Legion::FieldSpace field_space, Legion::FieldID field_id);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;
  void record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const override;

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

// ==========================================================================================

class ScalarStoreArg final : public AnalyzableBase {
 public:
  ScalarStoreArg(LogicalStore* store,
                 Legion::Future future,
                 std::size_t scalar_offset,
                 bool read_only,
                 GlobalRedopID redop);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;

 private:
  LogicalStore* store_{};
  Legion::Future future_{};
  std::size_t scalar_offset_{};
  bool read_only_{};
  GlobalRedopID redop_{};
};

class ReplicatedScalarStoreArg final : public AnalyzableBase {
 public:
  ReplicatedScalarStoreArg(LogicalStore* store,
                           Legion::FutureMap future_map,
                           std::size_t scalar_offset,
                           bool read_only);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;

 private:
  LogicalStore* store_{};
  Legion::FutureMap future_map_{};
  std::size_t scalar_offset_{};
  bool read_only_{};
};

class WriteOnlyScalarStoreArg final : public AnalyzableBase {
 public:
  WriteOnlyScalarStoreArg(LogicalStore* store, GlobalRedopID redop);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;

 private:
  LogicalStore* store_{};
  GlobalRedopID redop_{};
};

// ==========================================================================================

class BaseArrayArg final : public AnalyzableBase {
 public:
  BaseArrayArg(StoreAnalyzable data, std::optional<StoreAnalyzable> null_mask);

  explicit BaseArrayArg(StoreAnalyzable data);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  StoreAnalyzable data_;
  std::optional<StoreAnalyzable> null_mask_{};
};

class ListArrayArg final : public AnalyzableBase {
 public:
  ListArrayArg(InternalSharedPtr<Type> type,
               ArrayAnalyzable&& descriptor,
               ArrayAnalyzable&& vardata);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  InternalSharedPtr<Type> type_{};

  // We still need to pimpl these because ListArrayArg holds ArrayAnalyzable. And since
  // ArrayAnalyzable is a variant containing ListArrayArg, this leads to an infinite recursive
  // definition, that is similar to:
  //
  // class Foo {
  //   Foo f;
  // };
  //
  // Which is not allowed. We are able to remove the allocation for all other cases (and
  // ListArrayArg is rare anyways), so this one pimpl is a price worth paying.
  class Impl;

  std::unique_ptr<Impl> pimpl_;
};

class StructArrayArg final : public AnalyzableBase {
 public:
  StructArrayArg(InternalSharedPtr<Type> type,
                 std::optional<StoreAnalyzable> null_mask,
                 std::vector<ArrayAnalyzable>&& fields);

  void pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const override;
  void analyze(StoreAnalyzer& analyzer) const override;
  [[nodiscard]] std::optional<Legion::ProjectionID> get_key_proj_id() const override;
  void record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const override;
  void perform_invalidations() const override;

 private:
  InternalSharedPtr<Type> type_{};
  std::optional<StoreAnalyzable> null_mask_{};
  // This must stay as a vector because SmallVector tries to inspect sizeof(T), which at this
  // point would be incomplete
  std::vector<ArrayAnalyzable> fields_{};
};

class ListArrayArg::Impl {
 public:
  Impl(ArrayAnalyzable descr, ArrayAnalyzable var);

  ArrayAnalyzable descriptor;
  ArrayAnalyzable vardata;
};

}  // namespace legate::detail

#include <legate/operation/detail/launcher_arg.inl>
