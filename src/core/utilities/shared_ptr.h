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

#include "core/utilities/internal_shared_ptr.h"

namespace legate {

template <typename T>
class SharedPtr;

template <typename T>
void swap(SharedPtr<T>&, SharedPtr<T>&) noexcept;

template <typename T>
class SharedPtr {
  using internal_ptr_type = InternalSharedPtr<T>;

 public:
  using element_type   = typename internal_ptr_type::element_type;
  using ref_count_type = typename internal_ptr_type::ref_count_type;

  // Constructors
  constexpr SharedPtr() noexcept = default;

  SharedPtr(std::nullptr_t) noexcept;

  template <typename U, typename Deleter, typename Alloc = std::allocator<U>>
  SharedPtr(U* ptr, Deleter deleter, Alloc allocator = Alloc{});
  template <typename U>
  explicit SharedPtr(U* ptr);

  SharedPtr(const SharedPtr&) noexcept;
  SharedPtr& operator=(const SharedPtr&) noexcept;
  SharedPtr(SharedPtr&&) noexcept;
  SharedPtr& operator=(SharedPtr&&) noexcept;

  template <typename U>
  SharedPtr(const SharedPtr<U>&) noexcept;
  template <typename U>
  SharedPtr& operator=(const SharedPtr<U>&) noexcept;
  template <typename U>
  SharedPtr(SharedPtr<U>&&) noexcept;
  template <typename U>
  SharedPtr& operator=(SharedPtr<U>&&) noexcept;

  explicit SharedPtr(const InternalSharedPtr<element_type>&) noexcept;
  SharedPtr& operator=(const InternalSharedPtr<element_type>&) noexcept;
  explicit SharedPtr(InternalSharedPtr<element_type>&&) noexcept;
  SharedPtr& operator=(InternalSharedPtr<element_type>&&) noexcept;

  template <typename U>
  explicit SharedPtr(const InternalSharedPtr<U>&) noexcept;
  template <typename U>
  SharedPtr& operator=(const InternalSharedPtr<U>&) noexcept;
  template <typename U>
  explicit SharedPtr(InternalSharedPtr<U>&&) noexcept;
  template <typename U>
  SharedPtr& operator=(InternalSharedPtr<U>&&) noexcept;

  template <typename U, typename D>
  SharedPtr(std::unique_ptr<U, D>&&);
  template <typename U, typename D>
  SharedPtr& operator=(std::unique_ptr<U, D>&&);

  ~SharedPtr() noexcept;

  // Modifiers
  void swap(SharedPtr&) noexcept;
  // must namespace qualify to disambiguate from member function, otherwise clang balks
  friend void ::legate::swap<>(SharedPtr&, SharedPtr&) noexcept;
  void reset() noexcept;
  void reset(std::nullptr_t) noexcept;
  template <typename U, typename D = std::default_delete<U>, typename A = std::allocator<U>>
  void reset(U* ptr, D deleter = D{}, A allocator = A{});

  // Observers
  [[nodiscard]] element_type* get() const noexcept;
  [[nodiscard]] element_type& operator*() const noexcept;
  [[nodiscard]] element_type* operator->() const noexcept;

  [[nodiscard]] ref_count_type use_count() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct copy_tag {};
  struct move_tag {};

  template <typename U>
  SharedPtr(copy_tag, const InternalSharedPtr<U>& other) noexcept;
  template <typename U>
  SharedPtr(move_tag, InternalSharedPtr<U>&& other, bool from_internal_ptr) noexcept;
  template <typename U>
  void assign_(copy_tag, const InternalSharedPtr<U>&) noexcept;
  template <typename U>
  void assign_(move_tag, InternalSharedPtr<U>&&) noexcept;

  template <typename U>
  friend class SharedPtr;

  void reference_() noexcept;
  void dereference_() noexcept;

  internal_ptr_type ptr_{};
};

}  // namespace legate

#include "core/utilities/shared_ptr.inl"
