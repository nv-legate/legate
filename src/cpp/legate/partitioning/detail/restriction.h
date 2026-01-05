/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>

#include <cstdint>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace legate::detail {

class Partition;

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
  /*
   * @brief Create the exact set of restriction as dimension_restrictions and set if inversion is
   * required.
   *
   * @param dimension_restrictions The passed in set of restriction.
   * @param require_invertible Indicate if inversion is required, false by default.
   */
  // NOLINTBEGIN(google-explicit-constructor)
  Restrictions(SmallVector<Restriction> dimension_restrictions, bool require_invertible = false);
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

  /**
   * @brief Given a partition check if that satisfies this restriction.
   *
   * @param partition The partition to check.
   */
  [[nodiscard]] bool are_satisfied_by(const Partition& partition) const;

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
   * @param f The mapping function.
   */
  template <typename FUNC>
  [[nodiscard]] Restrictions map(FUNC&& f) &&;

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
                           std::int64_t>
  prune_dimensions(Span<const std::uint64_t> shape) const;

 private:
  [[nodiscard]] bool require_invertible_() const;

  [[nodiscard]] SmallVector<Restriction>& dimension_restrictions_();
  [[nodiscard]] const SmallVector<Restriction>& dimension_restrictions_() const;

  bool req_invertible_{};
  SmallVector<Restriction> dim_restrictions_{};
};

}  // namespace legate::detail

#include <legate/partitioning/detail/restriction.inl>
