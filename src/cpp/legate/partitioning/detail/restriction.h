/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/projection.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <legion.h>

#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace legate::detail {

class Partition;
class Shape;

/**
 * @brief Enum to describe partitioning preference on dimensions of a store
 */
enum class Restriction : std::uint8_t {
  ALLOW  = 0, /*!< The dimension can be partitioned */
  AVOID  = 1, /*!< The dimension can be partitioned, but other dimensions are preferred */
  FORBID = 2, /*!< The dimension must not be partitioned */
};

/**
 * @brief This class encapsulates the partitioning restriction information for all
 * dimensions of a store.
 *
 * The Restriction enums specify if there is any restriction to partition a store
 * on a certain dimension. The different states are explained in the Restriction class.
 */
class Restrictions {
 public:
  Restrictions() = default;
  /*
   * @brief Create a set of ALLOW restrction of ndim size.
   *
   * @param ndim The number of dimensions.
   */
  explicit Restrictions(std::uint32_t ndim);

  explicit Restrictions(const SmallVector<Restriction>& dimension_restrictions);
  /*
   * @brief Create the exact set of restriction as dimension_restrictions and set if inversion is
   * required.
   *
   * @param dimension_restrictions The passed in set of restriction.
   * @param require_invertible Indicate if inversion is required, false by default.
   */
  // NOLINTBEGIN(google-explicit-constructor)
  Restrictions(SmallVector<Restriction> dimension_restrictions,
               SmallVector<std::uint64_t, LEGATE_MAX_DIM> minimum_extents,
               bool require_invertible = false);
  // NOLINTEND(google-explicit-constructor)

  Restrictions(const Restrictions&)            = default;
  Restrictions& operator=(const Restrictions&) = default;

  Restrictions(Restrictions&&)            = default;
  Restrictions& operator=(Restrictions&&) = default;

  [[nodiscard]] bool operator==(const Restrictions& other) const;

  /**
   * @brief Set if the partition needs to be invertible or not.
   *
   * @param new_value Boolean to indicate if the partition needs to be invertible or not.
   */
  void set_require_invertible(bool new_value);

  /**
   * @brief Restrict partitioning in all dimensions.
   */
  void restrict_all_dimensions();
  /**
   * @brief Restrict partitioning in the provided dimension.
   *
   * @param dim The dimension we want to restrict partitioning on.
   */
  void restrict_dimension(std::uint32_t dim);

  void apply_minimum_extents(Span<const std::uint64_t> new_minimum_extents);

  /**
   * @brief Given a partition check if that satisfies this restriction.
   *
   * @param partition The partition to check.
   * @param shape The shape to partition
   */
  [[nodiscard]] bool are_satisfied_by(const Partition& partition,
                                      const InternalSharedPtr<Shape>& shape) const;

  [[nodiscard]] bool minimum_extents_satisfied_by(Span<const std::uint64_t> shape,
                                                  Span<const std::uint64_t> color_shape) const;

  /**
   * @brief Join another restrction object. This can be seen as an OR operation.
   *
   * @param other The other restrction to join.
   *
   * @return The joint restriction object.
   */
  [[nodiscard]] Restrictions join(const Restrictions& other) const;
  /**
   * @brief Join another restrction object but in place.
   *
   * @param other The other restrction to join.
   */
  void join_inplace(const Restrictions& other);

  /**
   * @brief Map every dimension with the function provided.
   *
   * The function f has the following signature
   * SmallVector<Restriction> f (SmallVector<Restriction>)
   *
   * The function g has the following signature
   * SmallVector<std::uint64_t, LEGATE_MAX_DIM> f (SmallVector<std::uint64_t, LEGATE_MAX_DIM>)
   *
   * @param f The mapping function for dimension restrictions.
   * @param g The mapping function for minimum extents.
   */
  template <typename RES_FUNC, typename EXT_FUNC>
  [[nodiscard]] Restrictions map(RES_FUNC&& f, EXT_FUNC&& g) &&;

  /**
   * @brief Prune the dimensions that are of extent 1 or are not allowed
   *        to be be partitioned.
   *
   * @param shape The shape we'll prune the dimensions on.
   *
   * @return Tuple of pruned shape, dimensions and total volume.
   */
  [[nodiscard]] std::tuple<SmallVector<std::size_t, LEGATE_MAX_DIM>,
                           SmallVector<std::uint32_t, LEGATE_MAX_DIM>,
                           std::uint64_t>
  prune_dimensions(Span<const std::uint64_t> shape) const;

  /**
   * @brief Prune the dimensions of a Domain that are restricted by this object.
   *
   * The i-th dimension of the input domain will not appear in the output if dim_restrictions_[i] is
   * FORBID.
   *
   * @return The domain with FORBIDden dimensions pruned out.
   */
  [[nodiscard]] Legion::Domain prune_dimensions(const Legion::Domain& domain) const;

  [[nodiscard]] Legion::Domain embed(const Legion::Domain& domain) const;

  [[nodiscard]] std::size_t count_restricted() const;

  [[nodiscard]] SymbolicPoint to_projection() const;

 private:
  [[nodiscard]] bool require_invertible_() const;

  [[nodiscard]] SmallVector<Restriction>& dimension_restrictions_();
  [[nodiscard]] const SmallVector<Restriction>& dimension_restrictions_() const;

  [[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM>& minimum_extents_();
  [[nodiscard]] const SmallVector<std::uint64_t, LEGATE_MAX_DIM>& minimum_extents_() const;

  void cache_needs_minimum_extent_check_();

  bool needs_minimum_extent_check_{};
  bool req_invertible_{};
  SmallVector<Restriction> dim_restrictions_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> min_extents_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/restriction.inl>
