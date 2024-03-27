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

#include "core/utilities/detail/type_traits.h"
#include "core/utilities/internal_shared_ptr.h"

namespace legate {

template <typename T>
class SharedPtr;

template <typename T>
void swap(SharedPtr<T>&, SharedPtr<T>&) noexcept;

template <typename T>
class SharedPtr {
 public:
  using internal_ptr_type = InternalSharedPtr<T>;
  using element_type      = typename internal_ptr_type::element_type;
  using ref_count_type    = typename internal_ptr_type::ref_count_type;

  // Constructors
  constexpr SharedPtr() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor) to mimick std::shared_ptr ctor
  SharedPtr(std::nullptr_t) noexcept;

  template <typename U,
            typename Deleter,
            typename Alloc = std::allocator<U>,
            typename       = std::enable_if_t<traits::detail::ptr_compat_v<U, element_type>>>
  SharedPtr(U* ptr, Deleter deleter, Alloc allocator = Alloc{});
  template <typename U, typename = std::enable_if_t<traits::detail::ptr_compat_v<U, element_type>>>
  explicit SharedPtr(U* ptr);

  SharedPtr(const SharedPtr&) noexcept;
  SharedPtr& operator=(const SharedPtr&) noexcept;
  SharedPtr(SharedPtr&&) noexcept;
  SharedPtr& operator=(SharedPtr&&) noexcept;

  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  SharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    const SharedPtr<U>&) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  SharedPtr& operator=(const SharedPtr<U>&) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  SharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    SharedPtr<U>&&) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  SharedPtr& operator=(SharedPtr<U>&&) noexcept;

  explicit SharedPtr(const InternalSharedPtr<element_type>&) noexcept;
  SharedPtr& operator=(const InternalSharedPtr<element_type>&) noexcept;
  explicit SharedPtr(InternalSharedPtr<element_type>&&) noexcept;
  SharedPtr& operator=(InternalSharedPtr<element_type>&&) noexcept;

  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  explicit SharedPtr(const InternalSharedPtr<U>&) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  SharedPtr& operator=(const InternalSharedPtr<U>&) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  explicit SharedPtr(InternalSharedPtr<U>&&) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  SharedPtr& operator=(InternalSharedPtr<U>&&) noexcept;

  template <
    typename U,
    typename D,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename std::unique_ptr<U, D>::element_type, element_type>>>
  SharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    std::unique_ptr<U, D>&&);
  template <
    typename U,
    typename D,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename std::unique_ptr<U, D>::element_type, element_type>>>
  SharedPtr& operator=(std::unique_ptr<U, D>&&);

  ~SharedPtr() noexcept;

  // Modifiers
  void swap(SharedPtr&) noexcept;
  // must namespace qualify to disambiguate from member function, otherwise clang balks
  friend void ::legate::swap<>(SharedPtr&, SharedPtr&) noexcept;
  void reset() noexcept;
  void reset(std::nullptr_t) noexcept;
  template <typename U,
            typename D = std::default_delete<U>,
            typename A = std::allocator<U>,
            typename   = std::enable_if_t<traits::detail::ptr_compat_v<U, element_type>>>
  void reset(U* ptr, D deleter = D{}, A allocator = A{});

  // Observers
  [[nodiscard]] element_type* get() const noexcept;
  [[nodiscard]] element_type& operator*() const noexcept;
  [[nodiscard]] element_type* operator->() const noexcept;

  [[nodiscard]] ref_count_type use_count() const noexcept;
  explicit operator bool() const noexcept;

  class InternalSharedPtrAccessTag {
    InternalSharedPtrAccessTag() = default;

    friend class SharedPtr<T>;
    friend class InternalSharedPtr<T>;
  };

  [[nodiscard]] internal_ptr_type& internal_ptr(InternalSharedPtrAccessTag) noexcept;
  [[nodiscard]] const internal_ptr_type& internal_ptr(InternalSharedPtrAccessTag) const noexcept;

 private:
  struct copy_tag {};
  struct move_tag {};

  template <typename U>
  SharedPtr(copy_tag, const InternalSharedPtr<U>& other) noexcept;
  template <typename U>
  SharedPtr(move_tag, InternalSharedPtr<U>&& other, bool from_internal_ptr) noexcept;

  template <typename U>
  friend class SharedPtr;

  void reference_() noexcept;
  void dereference_() noexcept;

  internal_ptr_type ptr_{};
};

// ==========================================================================================

template <typename T, typename U>
bool operator==(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator!=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator<(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator>(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator<=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator>=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept;

// ==========================================================================================

template <typename T>
bool operator==(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator==(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator!=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator!=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator<(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator<(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator>(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator>(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator<=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator<=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator>=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator>=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept;

}  // namespace legate

namespace std {

template <typename T>
struct hash<legate::SharedPtr<T>> {  // NOLINT(cert-dcl58-cpp) extending std::hash is OK
  [[nodiscard]] std::size_t operator()(const legate::SharedPtr<T>& ptr) const noexcept;
};

}  // namespace std

#include "core/utilities/shared_ptr.inl"
