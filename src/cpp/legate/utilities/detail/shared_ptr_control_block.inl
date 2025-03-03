/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/shared_ptr_control_block.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace legate::detail {

inline typename ControlBlockBase::ref_count_type ControlBlockBase::load_refcount_(
  const std::atomic<ref_count_type>& refcount) noexcept
{
  return refcount.load(std::memory_order_relaxed);
}

// GCC 13.2 generates error for -Wstringop-overflow in release build. Most likely a
// bug in GCC.  Detail: GCC thinks that the control block ptr for InternalSharedPtr
// can be null , and in error logs, legate::InternalSharedPtr<T>::maybe_destroy_()
// seems to be at the root, however all calls to maybe_destroy_() are from inside
// an `if (ctrl) {...}`, which should allow the compiler to deduce that `ctrl` is
// never null when calling maybe_destroy_()
//
// Sample error log below:
// In member function ‘std::__atomic_base<_IntTp>::__int_type
// std::__atomic_base<_IntTp>::fetch_add(__int_type, std::memory_order) [with _ITp
// = unsigned int]’,
// inlined from ‘static legate::detail::ControlBlockBase::ref_count_type
// legate::detail::ControlBlockBase::increment_refcount_(std::atomic<unsigned
// int>*)’ at
// /path/to/legate/utilities/detail/shared_ptr_control_block.inl:33:29,
// ...
// /usr/include/c++/13/bits/atomic_base.h:635:34: error: ‘unsigned int
// __atomic_add_fetch_4(volatile void*, unsigned int, int)’ writing 4 bytes into a
// region of size 0 overflows the destination [-Werror=stringop-overflow=]
// 635 |       { return __atomic_fetch_add(&_M_i, __i, int(__m)); }
// | ~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~
// In function ‘void legate::InternalSharedPtr<T>::init_shared_from_this_(const
// legate::EnableSharedFromThis<U>*, V*) [with U =
// legate::detail::LogicalRegionField; V = legate::detail::LogicalRegionField; T =
// legate::detail::LogicalRegionField]’:
//
// cc1plus: note: destination object is likely at address zero
//
LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_GCC_IGNORE("-Wstringop-overflow");
inline typename ControlBlockBase::ref_count_type ControlBlockBase::increment_refcount_(
  std::atomic<ref_count_type>* refcount) noexcept
{
  return refcount->fetch_add(1, std::memory_order_relaxed) + 1;
}
LEGATE_PRAGMA_POP();

inline typename ControlBlockBase::ref_count_type ControlBlockBase::decrement_refcount_(
  std::atomic<ref_count_type>* refcount) noexcept
{
  const auto v = refcount->fetch_sub(1, std::memory_order_release);

  LEGATE_ASSERT(v > 0);
  return v - 1;
}

inline void ControlBlockBase::maybe_destroy_control_block() noexcept
{
  if (!strong_ref_cnt() && !weak_ref_cnt() && !user_ref_cnt()) {
    // Refcount subs are std::memory_order_release, so all reads/writes to the refcounts are
    // completed at this point. Thus, if we happen to reach this point then we are certain that
    // we are the only remaining holder of the shared pointer.
    //
    // However, we still need to ensure the compiler does not reorder the call to
    // destroy_control_block() before our refcounts (because we do not "acquire" the resource
    // during the decrement).
    //
    // See https://stackoverflow.com/a/48148318/13615317
    std::atomic_thread_fence(std::memory_order_acquire);
    destroy_control_block();
  }
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_ref_cnt() const noexcept
{
  return load_refcount_(strong_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::weak_ref_cnt() const noexcept
{
  return load_refcount_(weak_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_ref_cnt() const noexcept
{
  return load_refcount_(user_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_ref() noexcept
{
  return increment_refcount_(&strong_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::weak_ref() noexcept
{
  return increment_refcount_(&weak_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_ref() noexcept
{
  return increment_refcount_(&user_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_deref() noexcept
{
  return decrement_refcount_(&strong_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::weak_deref() noexcept
{
  return decrement_refcount_(&weak_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_deref() noexcept
{
  return decrement_refcount_(&user_refs_);
}

// ==========================================================================================

template <typename T>
void ControlBlockBase::destroy_control_block_impl_(T* cb_impl) noexcept
{
  // This entire function is not allowed to throw
  auto alloc         = cb_impl->template rebind_alloc<T>();
  using alloc_traits = std::allocator_traits<std::decay_t<decltype(alloc)>>;

  // cb_impl->~T();
  alloc_traits::destroy(alloc, cb_impl);
  // operator delete(cb_impl);
  alloc_traits::deallocate(alloc, cb_impl, 1);
}

// ==========================================================================================

template <typename T, typename D, typename A>
typename SeparateControlBlock<T, D, A>::allocator_type&
SeparateControlBlock<T, D, A>::alloc_() noexcept
{
  return pair_.second();
}

template <typename T, typename D, typename A>
const typename SeparateControlBlock<T, D, A>::allocator_type&
SeparateControlBlock<T, D, A>::alloc_() const noexcept
{
  return pair_.second();
}

template <typename T, typename D, typename A>
typename SeparateControlBlock<T, D, A>::deleter_type&
SeparateControlBlock<T, D, A>::deleter_() noexcept
{
  return pair_.first();
}

template <typename T, typename D, typename A>
const typename SeparateControlBlock<T, D, A>::deleter_type&
SeparateControlBlock<T, D, A>::deleter_() const noexcept
{
  return pair_.first();
}

// ==========================================================================================

template <typename T, typename D, typename A>
SeparateControlBlock<T, D, A>::SeparateControlBlock(value_type* ptr,
                                                    deleter_type deleter,
                                                    allocator_type allocator) noexcept
  : ptr_{ptr}, pair_{std::move(deleter), std::move(allocator)}
{
  static_assert(
    std::is_nothrow_move_constructible_v<deleter_type>,
    "Deleter must be no-throw move constructible to preserve strong exception guarantee");
  static_assert(
    std::is_nothrow_move_constructible_v<allocator_type>,
    "Allocator must be no-throw move constructible to preserve strong exception guarantee");
}

template <typename T, typename D, typename A>
void SeparateControlBlock<T, D, A>::destroy_object() noexcept
{
  // See discussion in maybe_destroy_control_block() on why this memory fence is required.
  std::atomic_thread_fence(std::memory_order_acquire);
  // NOLINTNEXTLINE(bugprone-sizeof-expression): we want to compare with 0, that's the point
  static_assert(sizeof(value_type) > 0, "Value type must be complete at destruction");
  LEGATE_ASSERT(ptr_);
  deleter_()(ptr_);
}

template <typename T, typename D, typename A>
void SeparateControlBlock<T, D, A>::destroy_control_block() noexcept
{
  ControlBlockBase::destroy_control_block_impl_(this);
}

template <typename T, typename D, typename A>
void* SeparateControlBlock<T, D, A>::ptr() noexcept
{
  return static_cast<void*>(ptr_);
}

template <typename T, typename D, typename A>
const void* SeparateControlBlock<T, D, A>::ptr() const noexcept
{
  return static_cast<const void*>(ptr_);
}

template <typename T, typename D, typename A>
template <typename U>
auto SeparateControlBlock<T, D, A>::rebind_alloc() const
{
  using rebound_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<U>;
  return rebound_type{alloc_()};
}

// ==========================================================================================

template <typename T, typename A>
typename InplaceControlBlock<T, A>::allocator_type& InplaceControlBlock<T, A>::alloc_() noexcept
{
  return pair_.first();
}

template <typename T, typename A>
const typename InplaceControlBlock<T, A>::allocator_type& InplaceControlBlock<T, A>::alloc_()
  const noexcept
{
  return pair_.first();
}

template <typename T, typename A>
typename InplaceControlBlock<T, A>::AlignedStorage& InplaceControlBlock<T, A>::store_() noexcept
{
  return pair_.second();
}

template <typename T, typename A>
const typename InplaceControlBlock<T, A>::AlignedStorage& InplaceControlBlock<T, A>::store_()
  const noexcept
{
  return pair_.second();
}

// ==========================================================================================

template <typename T, typename A>
template <typename... Args>
InplaceControlBlock<T, A>::InplaceControlBlock(allocator_type allocator, Args&&... args)
  : pair_{std::move(allocator), nullptr}
{
  auto alloc = rebind_alloc<value_type>();

  // possibly throwing
  std::allocator_traits<std::decay_t<decltype(alloc)>>::construct(
    alloc, static_cast<value_type*>(ptr()), std::forward<Args>(args)...);
}

template <typename T, typename A>
void InplaceControlBlock<T, A>::destroy_object() noexcept
{
  // NOLINTNEXTLINE(bugprone-sizeof-expression): we want to compare with 0, that's the point
  static_assert(sizeof(value_type) > 0, "Value type must be complete at destruction");
  auto alloc = rebind_alloc<value_type>();

  std::allocator_traits<std::decay_t<decltype(alloc)>>::destroy(alloc,
                                                                static_cast<value_type*>(ptr()));
}

template <typename T, typename A>
void InplaceControlBlock<T, A>::destroy_control_block() noexcept
{
  ControlBlockBase::destroy_control_block_impl_(this);
}

template <typename T, typename A>
void* InplaceControlBlock<T, A>::ptr() noexcept
{
  return store_().addr();
}

template <typename T, typename A>
const void* InplaceControlBlock<T, A>::ptr() const noexcept
{
  return store_().addr();
}

template <typename T, typename A>
template <typename U>
auto InplaceControlBlock<T, A>::rebind_alloc() const
{
  using rebound_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<U>;
  return rebound_type{alloc_()};
}

// ==========================================================================================

template <typename U, typename Alloc, typename P, typename... Args>
U* construct_from_allocator_(  // NOLINT(readability-identifier-naming)
  Alloc& allocator,
  P* hint,
  Args&&... args)
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

}  // namespace legate::detail
