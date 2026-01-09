/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <type_traits>

namespace legate::detail {

/**
 * @brief A specialized accessor policy for `std::mspan` that models a Legion reduction
 * accessor.
 *
 * @tparam Redop The Legion reduction operation.
 * @tparam Exclusive `true` if the reduction has exclusive access to the buffer, `false`
 * otherwise.
 *
 * Reduction accessors with "exclusive" access do not use atomics to update the
 * values. Non-exclusive reduction accessors use atomics.
 */
template <typename Redop, bool Exclusive>
class ReductionAccessor {
  class ReferenceWrapper;

 public:
  using reduction_type   = Redop;
  using element_type     = typename reduction_type::LHS;
  using reference        = ReferenceWrapper;
  using data_handle_type = element_type*;

  static constexpr auto EXCLUSIVE = Exclusive;

 private:
  /**
   * @brief A wrapper over a reference to an element which exposes only the Legion accessor
   * methods.
   */
  class ReferenceWrapper {
   public:
    /**
     * @brief Construct the reference wrapper.
     *
     * @param elem A pointer to the element.
     */
    constexpr explicit ReferenceWrapper(data_handle_type elem) noexcept;
    // Must define these so that copies don't match the const T& overload
    constexpr ReferenceWrapper(const ReferenceWrapper&) noexcept            = default;
    constexpr ReferenceWrapper& operator=(const ReferenceWrapper&) noexcept = default;
    constexpr ReferenceWrapper(ReferenceWrapper&&) noexcept                 = default;
    constexpr ReferenceWrapper& operator=(ReferenceWrapper&&) noexcept      = default;

    /**
     * @brief Assign a value to the element.
     *
     * @return A reference to this.
     *
     * No value is *ever* assigned as part of this operator. It is illegal to directly assign
     * values to reduction accessors since the underlying buffer might already be initialized
     * with the reduction results of a prior task launch.
     *
     * We could =delete the method, but then the error message is inscrutable, so we define
     * this operator in order to static_assert() in it to provide a better error message.
     */
    template <typename T>
    constexpr ReferenceWrapper& operator=(const T&) noexcept;

    /**
     * @brief Reduce a value into the buffer.
     *
     * @param val The value to apply the reduction with.
     */
    constexpr void reduce(const typename reduction_type::RHS& val) noexcept;

    /**
     * @brief Reduce a value into the buffer.
     *
     * @param val The value to apply the reduction with.
     */
    constexpr void operator<<=(const typename reduction_type::RHS& val) noexcept;

    /**
     * @brief Read the current value.
     *
     * @return A const-reference to the current value if `Exclusive` is `true`, or a copy if
     * `Exclusive` if not.
     *
     * If `Exclusive` is `false`, the value is read atomically.
     */
    [[nodiscard]] constexpr std::conditional_t<Exclusive, const element_type&, element_type> get()
      const noexcept;

    /**
     * @return A pointer to the current value.
     *
     * This method may be used to perform the reduction operation using some external function,
     * such as CUDA's `atomicAdd()`. The user must take care to adhere to the rules of Legion
     * reduction accessor themselves when this method is used, as no additional protections are
     * afforded.
     */
    [[nodiscard]] constexpr data_handle_type data() const noexcept;

   private:
    data_handle_type elem_{};
  };

 public:
  constexpr ReductionAccessor() noexcept = default;

  template <
    typename U,
    bool UExcl,
    typename = std::enable_if_t<std::is_convertible_v<typename U::LHS (*)[], element_type (*)[]>>>
  constexpr ReductionAccessor(  // NOLINT(google-explicit-constructor)
    const ReductionAccessor<U, UExcl>&) noexcept;

  /**
   * @brief Access the specified element.
   *
   * @param p The data handle to access.
   * @param i The element number to access.
   *
   * @return A reference to the element.
   */
  [[nodiscard]] constexpr reference access(data_handle_type p, std::size_t i) const noexcept;

  /**
   * @param p The data handle to offset.
   * @param i The element number to offset by.
   *
   * @return The data handle advanced by `i`.
   */
  [[nodiscard]] constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept;
};

}  // namespace legate::detail

#include <legate/utilities/detail/mdspan/reduction_accessor.inl>
