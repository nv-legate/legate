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

#include "core/utilities/compressed_pair.h"
#include "core/utilities/internal_shared_ptr.h"

#include "legate_defines.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

namespace legate {

namespace detail {

// ==========================================================================================

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_ref_cnt() const noexcept
{
  return strong_refs_;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_ref_cnt() const noexcept
{
  return user_refs_;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_ref() noexcept
{
  return ++strong_refs_;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_ref() noexcept
{
  return ++user_refs_;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_deref() noexcept
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(strong_refs_ > 0);
  }
  return --strong_refs_;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_deref() noexcept
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(user_refs_ > 0);
  }
  return --user_refs_;
}

// ==========================================================================================

template <typename T>
inline void ControlBlockBase::dispose_control_block_impl_(T* cb_impl)
{
  auto alloc         = cb_impl->template rebind_alloc<T>();
  using alloc_traits = std::allocator_traits<std::decay_t<decltype(alloc)>>;

  // cb_impl->~T();
  alloc_traits::destroy(alloc, cb_impl);
  // operator delete(cb_impl);
  alloc_traits::deallocate(alloc, cb_impl, 1);
}

// ==========================================================================================

template <typename T, typename Deleter, typename Alloc>
class SeparateControlBlock final : public ControlBlockBase {
 public:
  using value_type     = T;
  using deleter_type   = Deleter;
  using allocator_type = Alloc;

  SeparateControlBlock() = delete;

  SeparateControlBlock(value_type* ptr, deleter_type deleter, allocator_type allocator) noexcept
    : ptr_{ptr}, pair_{std::move(deleter), std::move(allocator)}
  {
    static_assert(
      std::is_nothrow_move_constructible_v<deleter_type>,
      "Deleter must be no-throw move constructible to preserve strong exception guarantee");
    static_assert(
      std::is_nothrow_move_constructible_v<allocator_type>,
      "Allocator must be no-throw move constructible to preserve strong exception guarantee");
  }

  void destroy() noexcept final
  {
    // NOLINTNEXTLINE(bugprone-sizeof-expression): we want to compare with 0, that's the point
    static_assert(sizeof(value_type) > 0, "Value type must be complete at destruction");
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(ptr_);
    }
    deleter_()(ptr_);
  }

  void dispose_control_block() noexcept final
  {
    ControlBlockBase::dispose_control_block_impl_(this);
  }

  [[nodiscard]] void* ptr() noexcept final { return static_cast<void*>(ptr_); }
  [[nodiscard]] const void* ptr() const noexcept final { return static_cast<const void*>(ptr_); }

  template <typename U>
  [[nodiscard]] auto rebind_alloc() const
  {
    using rebound_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<U>;
    return rebound_type{alloc_()};
  }

 private:
  [[nodiscard]] allocator_type& alloc_() noexcept { return pair_.second(); }
  [[nodiscard]] const allocator_type& alloc_() const noexcept { return pair_.second(); }

  [[nodiscard]] deleter_type& deleter_() noexcept { return pair_.first(); }
  [[nodiscard]] const deleter_type& deleter_() const noexcept { return pair_.first(); }

  value_type* ptr_;
  compressed_pair<deleter_type, allocator_type> pair_;
};

template <typename T, typename Allocator>
class InplaceControlBlock final : public ControlBlockBase {
 public:
  using value_type     = T;
  using allocator_type = Allocator;

 private:
  struct aligned_storage {
    constexpr aligned_storage() noexcept = default;
    // use this ctor to avoid zero-initializing the array
    constexpr aligned_storage(std::nullptr_t) noexcept {}

    [[nodiscard]] void* addr() noexcept { return static_cast<void*>(&mem); }
    [[nodiscard]] const void* addr() const noexcept { return static_cast<const void*>(&mem); }

    alignas(alignof(value_type)) std::byte mem[sizeof(value_type)];
  };

 public:
  InplaceControlBlock() = delete;

  template <typename... Args>
  InplaceControlBlock(allocator_type allocator, Args&&... args)
    : pair_{std::move(allocator), nullptr}
  {
    auto alloc = rebind_alloc<value_type>();

    // possibly throwing
    std::allocator_traits<std::decay_t<decltype(alloc)>>::construct(
      alloc, static_cast<value_type*>(ptr()), std::forward<Args>(args)...);
  }

  void destroy() noexcept final
  {
    // NOLINTNEXTLINE(bugprone-sizeof-expression): we want to compare with 0, that's the point
    static_assert(sizeof(value_type) > 0, "Value type must be complete at destruction");
    auto alloc = rebind_alloc<value_type>();

    std::allocator_traits<std::decay_t<decltype(alloc)>>::destroy(alloc,
                                                                  static_cast<value_type*>(ptr()));
  }

  void dispose_control_block() noexcept final
  {
    ControlBlockBase::dispose_control_block_impl_(this);
  }

  [[nodiscard]] void* ptr() noexcept final { return store_().addr(); }
  [[nodiscard]] const void* ptr() const noexcept final { return store_().addr(); }

  template <typename U>
  [[nodiscard]] auto rebind_alloc() const
  {
    using rebound_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<U>;
    return rebound_type{alloc_()};
  }

 private:
  [[nodiscard]] allocator_type& alloc_() noexcept { return pair_.first(); }
  [[nodiscard]] const allocator_type& alloc_() const noexcept { return pair_.first(); }

  [[nodiscard]] aligned_storage& store_() noexcept { return pair_.second(); }
  [[nodiscard]] const aligned_storage& store_() const noexcept { return pair_.second(); }

  compressed_pair<allocator_type, aligned_storage> pair_;
};

template <typename U, typename Alloc, typename P, typename... Args>
inline U* construct_from_allocator_(Alloc& allocator, P* hint, Args&&... args)
{
  using rebound_type   = typename std::allocator_traits<Alloc>::template rebind_alloc<U>;
  using rebound_traits = std::allocator_traits<rebound_type>;
  rebound_type rebound_alloc{allocator};

  // Don't cast result to U * (implicitly or explicitly). It is an error for the rebound
  // allocator to allocate anything other than U *, so we should catch that.
  auto result = rebound_traits::allocate(rebound_alloc, 1, hint);
  static_assert(std::is_same_v<std::decay_t<decltype(result)>, U*>);
  // OK if the preceding lines throw, it is assumed the allocator cleans up after itself
  // properly
  try {
    rebound_traits::construct(rebound_alloc, result, std::forward<Args>(args)...);
  } catch (...) {
    rebound_traits::deallocate(rebound_alloc, result, 1);
    throw;
  }
  return result;
}

}  // namespace detail

// ==========================================================================================

template <typename T>
void InternalSharedPtr<T>::maybe_destroy_() noexcept
{
  if (!use_count()) {
    ctrl_->destroy();
    // Do NOT delete, move, re-order, or otherwise modify the following lines under ANY
    // circumstances.
    //
    // They must stay exactly as they are. ctrl_->dispose_control_block() calls the moral
    // equivalent of "delete this", and hence any modification of ctrl_ hereafter is strictly
    // undefined behavior.
    //
    // BEGIN DO NOT MODIFY
    ctrl_->dispose_control_block();
    ctrl_ = nullptr;
    ptr_  = nullptr;
    // END DO NOT MODIFY
  }
}

template <typename T>
void InternalSharedPtr<T>::strong_reference_() noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
      assert(use_count());
    }
    ctrl_->strong_ref();
  }
}

template <typename T>
void InternalSharedPtr<T>::strong_dereference_() noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
    }
    ctrl_->strong_deref();
    maybe_destroy_();
  }
}

template <typename T>
void InternalSharedPtr<T>::user_reference_(SharedPtrAccessTag) noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
      assert(use_count());
    }
    ctrl_->user_ref();
  }
}

template <typename T>
void InternalSharedPtr<T>::user_dereference_(SharedPtrAccessTag) noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
    }
    ctrl_->user_deref();
    maybe_destroy_();
  }
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>::InternalSharedPtr(AllocateSharedTag,
                                               control_block_type* ctrl_impl,
                                               U* ptr) noexcept
  : ctrl_{ctrl_impl}, ptr_{ptr}
{
}

// ==========================================================================================

template <typename T>
inline InternalSharedPtr<T>::InternalSharedPtr(std::nullptr_t) noexcept
{
}

// clang-format off
template <typename T>
template <typename U, typename D, typename A>
inline InternalSharedPtr<T>::InternalSharedPtr(U* ptr, D deleter, A allocator) try :
  InternalSharedPtr{
    AllocateSharedTag{},
    detail::construct_from_allocator_<detail::SeparateControlBlock<U, D, A>>(allocator, ptr, ptr, deleter, allocator),
    ptr
  }
{
}
catch (...)
{
  deleter(ptr);
  throw;
}
// clang-format on

template <typename T>
inline InternalSharedPtr<T>::InternalSharedPtr(element_type* ptr)
  : InternalSharedPtr{ptr, std::default_delete<element_type>{}}
{
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>::InternalSharedPtr(U* ptr)
  : InternalSharedPtr{ptr, std::default_delete<U>{}}
{
}

template <typename T>
inline InternalSharedPtr<T>::InternalSharedPtr(const InternalSharedPtr& other) noexcept
  : InternalSharedPtr{AllocateSharedTag{}, other.ctrl_, other.ptr_}
{
  strong_reference_();
}

template <typename T>
inline InternalSharedPtr<T>&
InternalSharedPtr<T>::operator=(  // NOLINT(bugprone-unhandled-self-assignment): yes it does
  const InternalSharedPtr& other) noexcept
{
  InternalSharedPtr tmp{other};

  swap(tmp);
  return *this;
}

template <typename T>
inline InternalSharedPtr<T>::InternalSharedPtr(InternalSharedPtr&& other) noexcept
  : InternalSharedPtr{
      AllocateSharedTag{}, std::exchange(other.ctrl_, nullptr), std::exchange(other.ptr_, nullptr)}
{
  // we do not increment ref-counts here, the other pointer gives up ownership entirely (so
  // refcounts stay at previous levels)
}

template <typename T>
inline InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(InternalSharedPtr&& other) noexcept
{
  InternalSharedPtr tmp{std::move(other)};

  swap(tmp);
  return *this;
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>::InternalSharedPtr(const InternalSharedPtr<U>& other) noexcept
  : InternalSharedPtr{AllocateSharedTag{}, other.ctrl_, other.ptr_}
{
  strong_reference_();
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(
  const InternalSharedPtr<U>& other) noexcept
{
  InternalSharedPtr tmp{other};

  swap(tmp);
  return *this;
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>::InternalSharedPtr(InternalSharedPtr<U>&& other) noexcept
  : InternalSharedPtr{
      AllocateSharedTag{}, std::exchange(other.ctrl_, nullptr), std::exchange(other.ptr_, nullptr)}
{
}

template <typename T>
template <typename U>
inline InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(InternalSharedPtr<U>&& other) noexcept
{
  InternalSharedPtr tmp{std::move(other)};

  swap(tmp);
  return *this;
}

template <typename T>
template <typename U, typename D>
inline InternalSharedPtr<T>::InternalSharedPtr(std::unique_ptr<U, D>&& ptr)
  : InternalSharedPtr{ptr.get(), std::move(ptr.get_deleter())}
{
  // Release only after we have fully constructed ourselves to preserve strong exception
  // guarantee. If the above constructor throws, this has no effect.
  ptr.release();
}

template <typename T>
template <typename U, typename D>
inline InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(std::unique_ptr<U, D>&& ptr)
{
  InternalSharedPtr{std::move(ptr)}.swap(*this);
  return *this;
}

template <typename T>
inline InternalSharedPtr<T>::~InternalSharedPtr() noexcept
{
  strong_dereference_();
}

// ==========================================================================================

template <typename T>
inline void InternalSharedPtr<T>::swap(InternalSharedPtr& other) noexcept
{
  using std::swap;

  swap(other.ctrl_, ctrl_);
  swap(other.ptr_, ptr_);
}

template <typename T>
inline void swap(InternalSharedPtr<T>& lhs, InternalSharedPtr<T>& rhs) noexcept
{
  lhs.swap(rhs);
}

template <typename T>
inline void InternalSharedPtr<T>::reset() noexcept
{
  InternalSharedPtr<T>{}.swap(*this);
}

template <typename T>
inline void InternalSharedPtr<T>::reset(std::nullptr_t) noexcept
{
  reset();
}

template <typename T>
template <typename U, typename D, typename A>
inline void InternalSharedPtr<T>::reset(U* ptr, D deleter, A allocator)
{
  InternalSharedPtr<T>{ptr, std::move(deleter), std::move(allocator)}.swap(*this);
}

// ==========================================================================================

template <typename T>
inline typename InternalSharedPtr<T>::element_type* InternalSharedPtr<T>::get() const noexcept
{
  return ptr_;
}

template <typename T>
inline typename InternalSharedPtr<T>::element_type& InternalSharedPtr<T>::operator*() const noexcept
{
  return *ptr_;
}

template <typename T>
inline typename InternalSharedPtr<T>::element_type* InternalSharedPtr<T>::operator->()
  const noexcept
{
  return get();
}

template <typename T>
inline typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::use_count()
  const noexcept
{
  // SharedPtr's are a subset of InternalSharedPtr (since each one holds an InternalSharedPtr),
  // so the number of strong references gives the total unique references held to the pointer.
  return strong_ref_count();
}

template <typename T>
inline typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::strong_ref_count()
  const noexcept
{
  return ctrl_ ? ctrl_->strong_ref_cnt() : 0;
}

template <typename T>
inline typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::user_ref_count()
  const noexcept
{
  return ctrl_ ? ctrl_->user_ref_cnt() : 0;
}

template <typename T>
inline InternalSharedPtr<T>::operator bool() const noexcept
{
  return get() != nullptr;
}

// ==========================================================================================

template <typename T, typename... Args>
inline InternalSharedPtr<T> make_internal_shared(Args&&... args)
{
  using allocator_type = std::allocator<T>;

  auto alloc = allocator_type{};
  auto control_block =
    detail::construct_from_allocator_<detail::InplaceControlBlock<T, allocator_type>>(
      alloc, static_cast<T*>(nullptr), alloc, std::forward<Args>(args)...);
  return {typename InternalSharedPtr<T>::AllocateSharedTag{},
          control_block,
          static_cast<T*>(control_block->ptr())};
}

}  // namespace legate
