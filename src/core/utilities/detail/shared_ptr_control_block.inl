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

#include "legate_defines.h"

#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>

namespace legate::detail {

inline typename ControlBlockBase::ref_count_type ControlBlockBase::load_refcount_(
  const std::atomic<ref_count_type>& refcount) noexcept
{
  return refcount.load(std::memory_order_relaxed);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::increment_refcount_(
  std::atomic<ref_count_type>& refcount) noexcept
{
  return refcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::decrement_refcount_(
  std::atomic<ref_count_type>& refcount) noexcept
{
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(refcount > 0);
  }
  return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline void ControlBlockBase::maybe_destroy_control_block() noexcept
{
  if (!strong_ref_cnt() && !weak_ref_cnt() && !user_ref_cnt()) {
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
  return increment_refcount_(strong_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::weak_ref() noexcept
{
  return increment_refcount_(weak_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_ref() noexcept
{
  return increment_refcount_(user_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::strong_deref() noexcept
{
  return decrement_refcount_(strong_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::weak_deref() noexcept
{
  return decrement_refcount_(weak_refs_);
}

inline typename ControlBlockBase::ref_count_type ControlBlockBase::user_deref() noexcept
{
  return decrement_refcount_(user_refs_);
}

// ==========================================================================================

template <typename T>
void ControlBlockBase::destroy_control_block_impl_(T* cb_impl)
{
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
  // NOLINTNEXTLINE(bugprone-sizeof-expression): we want to compare with 0, that's the point
  static_assert(sizeof(value_type) > 0, "Value type must be complete at destruction");
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(ptr_);
  }
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
typename InplaceControlBlock<T, A>::aligned_storage& InplaceControlBlock<T, A>::store_() noexcept
{
  return pair_.second();
}

template <typename T, typename A>
const typename InplaceControlBlock<T, A>::aligned_storage& InplaceControlBlock<T, A>::store_()
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
U* construct_from_allocator_(Alloc& allocator, P* hint, Args&&... args)
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
