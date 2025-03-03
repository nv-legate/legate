/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <fmt/base.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <variant>
#include <vector>

namespace legate::detail {

class AutoTask;
class Task;

/**
 * @brief The private TaskSignature implementation.
 */
class TaskSignature {
 public:
  /**
   * @brief A type which encodes the number of inputs, outputs, reductions, or scalars that a
   * task may take.
   *
   * The variant has 2 forms:
   *
   * #. `std::uint32_t`: The task takes exactly `n` arguments of the given kind, no more, no
   *     less.
   * #. `std::pair<std::uint32_t, std::uint32_t>`: The task takes a (possibly
   *     unbounded) range of arguments. If `pair.second` is exactly
   *     `std::numeric_limits<std::uint32_t>::max()`, then the task must take at least
   *     `pair.first` but can take an unlimited number of arguments past that. Otherwise, the
   *     task takes `[pair.first, pair.second)` arguments.
   */
  class Nargs {
   public:
    /**
     * @brief Default-construct an `Nargs` object.
     *
     * By default it holds a value of 0.
     */
    constexpr Nargs() = default;

    /**
     * @brief Construct a single-valued `Nargs` object.
     *
     * @param value The value to hold.
     */
    explicit Nargs(std::uint32_t value);

    /**
     * @brief Construct a ranged-value `Nargs` object.
     *
     * @param lower The lower bound of the range of values to hold.
     * @param upper The upper bound of the range of values to hold.
     *
     * @throw std::out_of_range If `lower` <= `upper`.
     */
    Nargs(std::uint32_t lower, std::uint32_t upper);

    /**
     * @brief Get the upper limit of the value held by this Nargs object.
     *
     * The upper limit is defined as:
     *
     * - If `this` holds a single value, then it is that value.
     * - If `this` holds a range, then if `range.second` holds a value, it is that. Otherwise,
     *   it is `range.first`.
     *
     * The main use-case for this routine is to determine a size to pre-allocate storage. This
     * is why it returns `range.first` if the range is unbounded rather than
     * e.g. `std::numeric_limits::max()`.
     *
     * @return The upper limit of the range of values.
     */
    [[nodiscard]] std::uint32_t upper_limit() const;

    /**
     * @return A reference to the stored value.
     */
    [[nodiscard]] const std::variant<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>>&
    value() const;

    /**
     * @brief Determine if `size` is compatible with this `Nargs` object.
     *
     * This routine may be used to check whether a given number of arguments matches the
     * expected `Nargs`. For example, given some array `inputs`, then
     * `nargs.compatible_with(inputs.size())` will return `true` if the number of inputs
     * matches the spec.
     *
     * If `strict` is `true` and `this` holds a single value, then `size` must match the stored
     * value exactly. If `strict` is `false`, then `size` must be <= the stored value.
     *
     * `strict` has no effect on the ranged variant, which always returns `range.first <= size
     * < range.second`.
     *
     * @param size The value to check.
     * @param strict Whether strict compatibility is needed.
     *
     * @return `true` if `size` is compatible, `false` otherwise.
     */
    [[nodiscard]] bool compatible_with(std::size_t size, bool strict = true) const;

   private:
    std::variant<std::uint32_t, std::pair<std::uint32_t, std::uint32_t>> value_{};
  };

  constexpr void inputs(std::optional<Nargs> n) noexcept;
  constexpr void outputs(std::optional<Nargs> n) noexcept;
  constexpr void scalars(std::optional<Nargs> n) noexcept;
  constexpr void redops(std::optional<Nargs> n) noexcept;
  void constraints(
    std::optional<std::vector<InternalSharedPtr<detail::ProxyConstraint>>> cstrnts) noexcept;

  [[nodiscard]] constexpr const std::optional<Nargs>& inputs() const noexcept;
  [[nodiscard]] constexpr const std::optional<Nargs>& outputs() const noexcept;
  [[nodiscard]] constexpr const std::optional<Nargs>& scalars() const noexcept;
  [[nodiscard]] constexpr const std::optional<Nargs>& redops() const noexcept;
  [[nodiscard]] std::optional<Span<const InternalSharedPtr<detail::ProxyConstraint>>> constraints()
    const noexcept;

  /**
   * @brief Validate that the signature is sane.
   *
   * For example, this call checks if the various indices into arrays match the expected
   * input/output/reduction signatures given. This can, of course, only be done if the
   * corresponding argument bounds have been supplied by the user.
   *
   * @param task_name The name of the task that this signature belongs to.
   */
  void validate(std::string_view task_name) const;

  /**
   * @brief Check that a task's signature matches the expected signature.
   *
   * @param task The task object to check.
   *
   * @throw std::out_of_range If the task arguments are not the proscribed size.
   */
  void check_signature(const Task& task) const;

  /**
   * @brief Apply constraints detailed in `this` on `task`.
   *
   * @param task The task to apply the constraints to.
   */
  void apply_constraints(AutoTask* task) const;

 private:
  std::optional<Nargs> num_inputs_{};
  std::optional<Nargs> num_outputs_{};
  std::optional<Nargs> num_scalars_{};
  std::optional<Nargs> num_redops_{};
  std::optional<std::vector<InternalSharedPtr<detail::ProxyConstraint>>> constraints_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_signature.inl>

namespace fmt {

template <>
struct formatter<legate::detail::TaskSignature::Nargs> : formatter<std::string> {
  format_context::iterator format(const legate::detail::TaskSignature::Nargs& nargs,
                                  format_context& ctx) const;
};

}  // namespace fmt
