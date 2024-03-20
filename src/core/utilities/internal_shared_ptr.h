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

#include "core/utilities/detail/shared_ptr_control_block.h"
#include "core/utilities/detail/type_traits.h"

#include <memory>
#include <string>
#include <type_traits>

namespace legate {

namespace detail {

// T is always the same T given as the shared pointer type. So we use U here because we want to
// delete the most derived type (which might not be T).
template <typename /* T */, typename U>
struct shared_ptr_default_delete : std::default_delete<U> {};

// Unless it is an array type, in which case we fall back to using T to deduce the deleter. U
// is always given as a pointer-type, not an array, and so we want to preserve the fact that
// the pointer is deleted via delete[].
template <typename T, typename U, std::size_t N>
struct shared_ptr_default_delete<T[N], U> : std::default_delete<T[]> {};

template <typename T, typename U>
struct shared_ptr_default_delete<T[], U> : std::default_delete<T[]> {};

}  // namespace detail

template <typename T>
class EnableSharedFromThis;

template <typename T>
class SharedPtr;

template <typename T>
class InternalSharedPtr;

template <typename T>
class InternalWeakPtr;

template <typename T>
void swap(InternalWeakPtr<T>&, InternalWeakPtr<T>&) noexcept;

template <typename T>
class InternalWeakPtr {
  using control_block_type = detail::ControlBlockBase;
  using shared_type        = InternalSharedPtr<T>;

 public:
  using element_type   = typename shared_type::element_type;
  using ref_count_type = typename control_block_type::ref_count_type;

  // Constructors
  constexpr InternalWeakPtr() noexcept = default;

  InternalWeakPtr(const InternalWeakPtr& other) noexcept;
  InternalWeakPtr& operator=(const InternalWeakPtr& other) noexcept;
  InternalWeakPtr(InternalWeakPtr&& other) noexcept;
  InternalWeakPtr& operator=(InternalWeakPtr&& other) noexcept;

  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalWeakPtr<U>::element_type, element_type>>>
  InternalWeakPtr(  // NOLINT(google-explicit-constructor) to mimick std::weak_ptr ctor
    const InternalWeakPtr<U>& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalWeakPtr<U>::element_type, element_type>>>
  InternalWeakPtr& operator=(const InternalWeakPtr<U>& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalWeakPtr<U>::element_type, element_type>>>
  InternalWeakPtr& operator=(InternalWeakPtr<U>&& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalWeakPtr<U>::element_type, element_type>>>
  InternalWeakPtr(  // NOLINT(google-explicit-constructor) to mimick std::weak_ptr ctor
    InternalWeakPtr<U>&& other) noexcept;

  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalWeakPtr(  // NOLINT(google-explicit-constructor) to mimick std::weak_ptr ctor
    const InternalSharedPtr<U>& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalWeakPtr& operator=(const InternalSharedPtr<U>& other) noexcept;

  ~InternalWeakPtr() noexcept;

  // Observers
  [[nodiscard]] ref_count_type use_count() const noexcept;
  [[nodiscard]] bool expired() const noexcept;

  // Getters
  // See definition of lock() for why we are silencing this
  [[nodiscard]] InternalSharedPtr<T> lock() const noexcept;  // NOLINT(bugprone-exception-escape)

  // Modifiers
  void swap(InternalWeakPtr& other) noexcept;
  friend void ::legate::swap<>(InternalWeakPtr& lhs, InternalWeakPtr& rhs) noexcept;

 private:
  template <typename U>
  friend class InternalSharedPtr;

  void maybe_destroy_() noexcept;
  void weak_reference_() noexcept;
  void weak_dereference_() noexcept;

  struct copy_tag {};
  struct move_tag {};

  constexpr InternalWeakPtr(move_tag, control_block_type* ctrl_block_) noexcept;
  InternalWeakPtr(copy_tag, control_block_type* ctrl_block_) noexcept;

  control_block_type* ctrl_{};
};

// ==========================================================================================

template <typename T>
void swap(InternalSharedPtr<T>&, InternalSharedPtr<T>&) noexcept;

template <typename T>
class InternalSharedPtr {
  using control_block_type = detail::ControlBlockBase;

 public:
  static_assert(!std::is_reference_v<T>);
  using element_type   = std::remove_extent_t<T>;
  using ref_count_type = typename control_block_type::ref_count_type;

  // constructors
  constexpr InternalSharedPtr() noexcept = default;

  // NOLINTNEXTLINE(google-explicit-constructor) to mimick std::shared_ptr ctor
  InternalSharedPtr(std::nullptr_t) noexcept;

  template <typename U,
            typename D,
            typename A = std::allocator<U>,
            typename   = std::enable_if_t<traits::detail::ptr_compat_v<U, element_type>>>
  InternalSharedPtr(U* ptr, D deleter, A allocator = A{});
  template <typename U, typename = std::enable_if_t<traits::detail::ptr_compat_v<U, T>>>
  explicit InternalSharedPtr(U* ptr);
  explicit InternalSharedPtr(element_type* ptr);

  InternalSharedPtr(const InternalSharedPtr& other) noexcept;
  // NOLINTNEXTLINE(bugprone-unhandled-self-assignment): yes it does
  InternalSharedPtr& operator=(const InternalSharedPtr& other) noexcept;
  InternalSharedPtr(InternalSharedPtr&& other) noexcept;
  // NOLINTNEXTLINE(bugprone-unhandled-self-assignment): yes it does
  InternalSharedPtr& operator=(InternalSharedPtr&& other) noexcept;

  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    const InternalSharedPtr<U>& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr& operator=(const InternalSharedPtr<U>& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    InternalSharedPtr<U>&& other) noexcept;
  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalSharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr& operator=(InternalSharedPtr<U>&& other) noexcept;

  template <
    typename U,
    typename D,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename std::unique_ptr<U, D>::element_type, element_type>>>
  InternalSharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    std::unique_ptr<U, D>&& ptr);
  template <
    typename U,
    typename D,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename std::unique_ptr<U, D>::element_type, element_type>>>
  InternalSharedPtr& operator=(std::unique_ptr<U, D>&& ptr);

  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    const SharedPtr<U>& other) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr& operator=(const SharedPtr<U>& other) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr(  // NOLINT(google-explicit-constructor) to mimick std::shared_ptr ctor
    SharedPtr<U>&& other) noexcept;
  template <typename U,
            typename = std::enable_if_t<
              traits::detail::ptr_compat_v<typename SharedPtr<U>::element_type, element_type>>>
  InternalSharedPtr& operator=(SharedPtr<U>&& other) noexcept;

  template <
    typename U,
    typename = std::enable_if_t<
      traits::detail::ptr_compat_v<typename InternalWeakPtr<U>::element_type, element_type>>>
  explicit InternalSharedPtr(const InternalWeakPtr<U>& other);

  // No SFINAE here, otherwise static_pointer_cast() and friends don't work!
  template <typename U>
  InternalSharedPtr(const InternalSharedPtr<U>& other, element_type* ptr) noexcept;
  template <typename U>
  InternalSharedPtr(InternalSharedPtr<U>&& other, element_type* ptr) noexcept;

  ~InternalSharedPtr() noexcept;

  // Modifiers
  void swap(InternalSharedPtr& other) noexcept;
  friend void ::legate::swap<>(InternalSharedPtr& lhs, InternalSharedPtr& rhs) noexcept;
  void reset() noexcept;
  void reset(std::nullptr_t) noexcept;
  template <typename U,
            typename D = detail::shared_ptr_default_delete<T, U>,
            typename A = std::allocator<U>,
            typename   = std::enable_if_t<traits::detail::ptr_compat_v<U, element_type>>>
  void reset(U* ptr, D deleter = D{}, A allocator = A{});

  // Observers
  [[nodiscard]] element_type& operator[](std::ptrdiff_t idx) noexcept;
  [[nodiscard]] const element_type& operator[](std::ptrdiff_t idx) const noexcept;
  [[nodiscard]] element_type* get() const noexcept;
  [[nodiscard]] element_type& operator*() const noexcept;
  [[nodiscard]] element_type* operator->() const noexcept;

  [[nodiscard]] ref_count_type use_count() const noexcept;
  [[nodiscard]] ref_count_type strong_ref_count() const noexcept;
  [[nodiscard]] ref_count_type user_ref_count() const noexcept;
  [[nodiscard]] ref_count_type weak_ref_count() const noexcept;
  explicit operator bool() const noexcept;

  [[nodiscard]] SharedPtr<T> as_user_ptr() const noexcept;

  class SharedPtrAccessTag {
    SharedPtrAccessTag() = default;

    friend class SharedPtr<T>;
    friend class InternalSharedPtr<T>;
  };

  void user_reference_(SharedPtrAccessTag) noexcept;
  void user_dereference_(SharedPtrAccessTag) noexcept;

 private:
  // Unfortunately we cannot just friend the make_internal_shared() for U = T, since that would
  // constitute a partial function specialization. So instead we just friend them all, because
  // friendship makes the world go 'round.
  template <typename U, typename... Args>
  friend InternalSharedPtr<U> make_internal_shared(Args&&... args);

  template <typename U>
  friend class InternalSharedPtr;

  template <typename U>
  friend class InternalWeakPtr;

  template <typename U, typename V>
  void init_shared_from_this_(const EnableSharedFromThis<U>*, V*);
  void init_shared_from_this_(...);

  void maybe_destroy_() noexcept;
  void strong_reference_() noexcept;
  void strong_dereference_() noexcept;
  void weak_reference_() noexcept;
  void weak_dereference_() noexcept;

  struct AllocatedControlBlockTag {};

  template <typename U>
  InternalSharedPtr(AllocatedControlBlockTag, control_block_type* ctrl_impl, U* ptr) noexcept;

  control_block_type* ctrl_{};
  element_type* ptr_{};
};

// ==========================================================================================

template <typename T, typename... Args>
[[nodiscard]] InternalSharedPtr<T> make_internal_shared(Args&&... args);

// ==========================================================================================

template <typename T, typename U>
[[nodiscard]] InternalSharedPtr<T> static_pointer_cast(const InternalSharedPtr<U>& ptr) noexcept;

template <typename T, typename U>
[[nodiscard]] InternalSharedPtr<T> static_pointer_cast(InternalSharedPtr<U>&& ptr) noexcept;

template <typename T, typename U>
[[nodiscard]] InternalSharedPtr<T> const_pointer_cast(const InternalSharedPtr<U>& ptr) noexcept;

template <typename T, typename U>
[[nodiscard]] InternalSharedPtr<T> const_pointer_cast(InternalSharedPtr<U>&& ptr) noexcept;

// ==========================================================================================

template <typename T, typename U>
bool operator==(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator!=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator<(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator>(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator<=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

template <typename T, typename U>
bool operator>=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept;

// ==========================================================================================

template <typename T>
bool operator==(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator==(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator!=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator!=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator<(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator<(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator>(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator>(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator<=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator<=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

template <typename T>
bool operator>=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept;

template <typename T>
bool operator>=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept;

// ==========================================================================================

template <typename T>
class EnableSharedFromThis {
 protected:
  constexpr EnableSharedFromThis() noexcept = default;
  constexpr EnableSharedFromThis(const EnableSharedFromThis&) noexcept;
  constexpr EnableSharedFromThis& operator=(const EnableSharedFromThis&) noexcept;

 public:
  using weak_type         = InternalWeakPtr<T>;
  using const_weak_type   = InternalWeakPtr<const T>;
  using shared_type       = InternalSharedPtr<T>;
  using const_shared_type = InternalSharedPtr<const T>;

  [[nodiscard]] shared_type shared_from_this();
  [[nodiscard]] const_shared_type shared_from_this() const;

 private:
  template <typename U>
  friend class InternalSharedPtr;

  mutable weak_type weak_this_{};
};

namespace detail {

template <typename T>
class shared_from_this_enabled {
  template <typename U>
  static constexpr auto test(U* u)
    -> decltype(static_cast<void>(u->shared_from_this()), std::true_type{});

  template <typename U>
  static constexpr std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(nullptr))::value;
};

template <typename T>
inline constexpr bool shared_from_this_enabled_v = shared_from_this_enabled<T>::value;

namespace shared_from_this_enabled_test {

struct NotEnabled {};

struct Enabled : EnableSharedFromThis<Enabled> {};

struct PrivateEnabled : private EnableSharedFromThis<PrivateEnabled> {};

struct DerivedEnabled : Enabled {};

struct DerivedPrivateEnabled : private Enabled {};

static_assert(!shared_from_this_enabled_v<NotEnabled>);
static_assert(shared_from_this_enabled_v<Enabled>);
static_assert(!shared_from_this_enabled_v<PrivateEnabled>);
static_assert(shared_from_this_enabled_v<DerivedEnabled>);
static_assert(!shared_from_this_enabled_v<DerivedPrivateEnabled>);

}  // namespace shared_from_this_enabled_test

}  // namespace detail

// ==========================================================================================

class BadInternalWeakPtr : public std::bad_weak_ptr {
 public:
  BadInternalWeakPtr() = default;

  explicit BadInternalWeakPtr(std::string what) noexcept;

  [[nodiscard]] const char* what() const noexcept override;

 private:
  std::string what_{};
};

}  // namespace legate

namespace std {

template <typename T>
struct hash<legate::InternalSharedPtr<T>> {  // NOLINT(cert-dcl58-cpp) extending std::hash is OK
  [[nodiscard]] std::size_t operator()(const legate::InternalSharedPtr<T>& ptr) const noexcept;
};

}  // namespace std

#include "core/utilities/internal_shared_ptr.inl"
