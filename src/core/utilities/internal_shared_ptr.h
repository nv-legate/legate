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

#include <cstdint>
#include <memory>
#include <type_traits>

namespace legate {

namespace detail {

class ControlBlockBase {
 public:
  using ref_count_type = uint32_t;

  constexpr ControlBlockBase() noexcept = default;
  virtual ~ControlBlockBase() noexcept  = default;

  virtual void destroy() noexcept                        = 0;
  virtual void dispose_control_block() noexcept          = 0;
  [[nodiscard]] virtual void* ptr() noexcept             = 0;
  [[nodiscard]] virtual const void* ptr() const noexcept = 0;

  [[nodiscard]] ref_count_type strong_ref_cnt() const noexcept;
  [[nodiscard]] ref_count_type user_ref_cnt() const noexcept;

  ref_count_type strong_ref() noexcept;
  ref_count_type user_ref() noexcept;
  ref_count_type strong_deref() noexcept;
  ref_count_type user_deref() noexcept;

 protected:
  template <typename T>
  static void dispose_control_block_impl_(T* cb_impl);

 private:
  ref_count_type strong_refs_{1};
  ref_count_type user_refs_{0};
};

}  // namespace detail

template <typename T>
class SharedPtr;

template <typename T>
class InternalSharedPtr;

template <typename T>
void swap(InternalSharedPtr<T>&, InternalSharedPtr<T>&) noexcept;

template <typename T, typename... Args>
[[nodiscard]] InternalSharedPtr<T> make_internal_shared(Args&&... args);

template <typename T>
class InternalSharedPtr {
  using control_block_type = detail::ControlBlockBase;

 public:
  using element_type   = std::remove_extent_t<T>;
  using ref_count_type = typename control_block_type::ref_count_type;
  static_assert(!std::is_reference_v<T>);

  // constructors
  constexpr InternalSharedPtr() noexcept = default;

  InternalSharedPtr(std::nullptr_t) noexcept;

  template <typename U, typename D, typename A = std::allocator<U>>
  InternalSharedPtr(U* ptr, D deleter, A allocator = A{});
  template <typename U>
  explicit InternalSharedPtr(U* ptr);
  explicit InternalSharedPtr(element_type* ptr);

  InternalSharedPtr(const InternalSharedPtr& other) noexcept;
  InternalSharedPtr& operator=(const InternalSharedPtr& other) noexcept;
  InternalSharedPtr(InternalSharedPtr&& other) noexcept;
  InternalSharedPtr& operator=(InternalSharedPtr&& other) noexcept;

  template <typename U>
  InternalSharedPtr(const InternalSharedPtr<U>& other) noexcept;
  template <typename U>
  InternalSharedPtr& operator=(const InternalSharedPtr<U>& other) noexcept;
  template <typename U>
  InternalSharedPtr(InternalSharedPtr<U>&& other) noexcept;
  template <typename U>
  InternalSharedPtr& operator=(InternalSharedPtr<U>&& other) noexcept;

  template <typename U, typename D>
  InternalSharedPtr(std::unique_ptr<U, D>&& ptr);
  template <typename U, typename D>
  InternalSharedPtr& operator=(std::unique_ptr<U, D>&& ptr);

  ~InternalSharedPtr() noexcept;

  // Modifiers
  void swap(InternalSharedPtr& other) noexcept;
  friend void ::legate::swap<>(InternalSharedPtr& lhs, InternalSharedPtr& rhs) noexcept;
  void reset() noexcept;
  void reset(std::nullptr_t) noexcept;
  template <typename U, typename D = std::default_delete<U>, typename A = std::allocator<U>>
  void reset(U* ptr, D deleter = D{}, A allocator = A{});

  // Observers
  [[nodiscard]] element_type* get() const noexcept;
  [[nodiscard]] element_type& operator*() const noexcept;
  [[nodiscard]] element_type* operator->() const noexcept;

  [[nodiscard]] ref_count_type use_count() const noexcept;
  [[nodiscard]] ref_count_type strong_ref_count() const noexcept;
  [[nodiscard]] ref_count_type user_ref_count() const noexcept;
  explicit operator bool() const noexcept;

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

  struct AllocateSharedTag {};

  template <typename U>
  InternalSharedPtr(AllocateSharedTag, control_block_type* ctrl_impl, U* ptr) noexcept;

  void maybe_destroy_() noexcept;
  void strong_reference_() noexcept;
  void strong_dereference_() noexcept;

  control_block_type* ctrl_{};
  element_type* ptr_{};
};

}  // namespace legate

#include "core/utilities/internal_shared_ptr.inl"
