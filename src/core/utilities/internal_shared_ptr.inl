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

#include "legate_defines.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <utility>

namespace legate {

template <typename T>
void InternalWeakPtr<T>::maybe_destroy_() noexcept
{
  if (ctrl_ && ctrl_->weak_ref_cnt() == 0) {
    // Do NOT delete, move, re-order, or otherwise modify the following lines under ANY
    // circumstances.
    //
    // They must stay exactly as they are. ctrl_->maybe_destroy_control_block() calls the moral
    // equivalent of "delete this", and hence any modification of ctrl_ hereafter is strictly
    // undefined behavior.
    //
    // We must null-out ctrl_ first (via std::exchange) since if this weak ptr is held by a
    // enable_shared_from_this, then maybe_destroy_control_block() could potentially also wipe
    // out *this.
    //
    // BEGIN DO NOT MODIFY
    std::exchange(ctrl_, nullptr)->maybe_destroy_control_block();
    // END DO NOT MODIFY
  }
}

template <typename T>
void InternalWeakPtr<T>::weak_reference_() noexcept
{
  if (ctrl_) {
    ctrl_->weak_ref();
  }
}

template <typename T>
void InternalWeakPtr<T>::weak_dereference_() noexcept
{
  if (ctrl_) {
    ctrl_->weak_deref();
    maybe_destroy_();
  }
}

template <typename T>
constexpr InternalWeakPtr<T>::InternalWeakPtr(move_tag, control_block_type* ctrl_block_) noexcept
  : ctrl_{ctrl_block_}
{
}

template <typename T>
InternalWeakPtr<T>::InternalWeakPtr(copy_tag, control_block_type* ctrl_block_) noexcept
  : ctrl_{ctrl_block_}
{
  weak_reference_();
}

// ==========================================================================================

template <typename T>
InternalWeakPtr<T>::InternalWeakPtr(const InternalWeakPtr& other) noexcept
  : InternalWeakPtr{copy_tag{}, other.ctrl_}
{
}

template <typename T>
InternalWeakPtr<T>&
InternalWeakPtr<T>::operator=(  // NOLINT(bugprone-unhandled-self-assignment) yes it does
  const InternalWeakPtr& other) noexcept
{
  InternalWeakPtr{other}.swap(*this);
  return *this;
}

template <typename T>
InternalWeakPtr<T>::InternalWeakPtr(InternalWeakPtr&& other) noexcept
  : InternalWeakPtr{move_tag{}, std::exchange(other.ctrl_, nullptr)}
{
}

template <typename T>
InternalWeakPtr<T>& InternalWeakPtr<T>::operator=(InternalWeakPtr&& other) noexcept
{
  InternalWeakPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>::InternalWeakPtr(const InternalWeakPtr<U>& other) noexcept
  : InternalWeakPtr{copy_tag{}, other.ctrl_}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>&
InternalWeakPtr<T>::operator=(  // NOLINT(bugprone-unhandled-self-assignment) yes it does
  const InternalWeakPtr<U>& other) noexcept
{
  InternalWeakPtr{other}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>::InternalWeakPtr(InternalWeakPtr<U>&& other) noexcept
  : InternalWeakPtr{move_tag{}, std::exchange(other.ctrl_, nullptr)}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>& InternalWeakPtr<T>::operator=(InternalWeakPtr<U>&& other) noexcept
{
  InternalWeakPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>::InternalWeakPtr(const InternalSharedPtr<U>& other) noexcept
  : InternalWeakPtr{copy_tag{}, other.ctrl_}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalWeakPtr<T>& InternalWeakPtr<T>::operator=(const InternalSharedPtr<U>& other) noexcept
{
  InternalWeakPtr{other}.swap(*this);
  return *this;
}

template <typename T>
InternalWeakPtr<T>::~InternalWeakPtr() noexcept
{
  weak_dereference_();
}

// ==========================================================================================

template <typename T>
typename InternalWeakPtr<T>::ref_count_type InternalWeakPtr<T>::use_count() const noexcept
{
  return ctrl_ ? ctrl_->strong_ref_cnt() : 0;
}

template <typename T>
bool InternalWeakPtr<T>::expired() const noexcept
{
  return use_count() == 0;
}

template <typename T>
InternalSharedPtr<T> InternalWeakPtr<T>::lock() const noexcept
{
  // Normally the weak ptr ctor for InternalSharedPtr can throw (if the weak ptr is empty) but
  // in this case know it is not, and hence this function is noexcept
  return expired() ? InternalSharedPtr<T>{} : InternalSharedPtr<T>{*this};
}

template <typename T>
void InternalWeakPtr<T>::swap(InternalWeakPtr& other) noexcept
{
  using std::swap;

  swap(other.ctrl_, ctrl_);
}

template <typename T>
void swap(InternalWeakPtr<T>& lhs, InternalWeakPtr<T>& rhs) noexcept
{
  lhs.swap(rhs);
}

// ==========================================================================================

template <typename T>
void InternalSharedPtr<T>::maybe_destroy_() noexcept
{
  if (use_count()) {
    return;
  }
  // If ptr_ is a SharedFromThis enabled class, then we want to temporarily act like we have an
  // extra strong ref. This is to head off the following:
  //
  // 1. We decrement strong refcount to 0 (i.e. we are here).
  // 2. Control block destroys the object (i.e. ptr_)...
  // 3. ... Which eventually calls the SharedFromThis dtor, which calls the weak ptr dtor...
  // 4. ... Which decrements the weak refcount to  0...
  // 5. ... And calls its own ctrl_->maybe_destroy_control_block()
  //
  // And since all 3 counts (strong, weak, and user) are 0, the control block self-destructs
  // out from underneath us! We could implement some complex logic which propagates a bool
  // "weak ptr don't destroy the control block" all the way down the stack somehow, or maybe
  // set a similar flag in the control block, or we can just do this.
  //
  // CAUTION:
  // If element_type is opaque at this point, it may constitute an ODR violation!
  if constexpr (detail::shared_from_this_enabled_v<element_type>) {
    ctrl_->strong_ref();
  }
  ctrl_->destroy_object();
  if constexpr (detail::shared_from_this_enabled_v<element_type>) {
    ctrl_->strong_deref();
  }
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!use_count());
  }
  // Do NOT delete, move, re-order, or otherwise modify the following lines under ANY
  // circumstances.
  //
  // They must stay exactly as they are. ctrl_->maybe_destroy_control_block() calls the moral
  // equivalent of "delete this", and hence any modification of ctrl_ hereafter is strictly
  // undefined behavior.
  //
  // BEGIN DO NOT MODIFY
  ctrl_->maybe_destroy_control_block();
  ctrl_ = nullptr;
  ptr_  = nullptr;
  // END DO NOT MODIFY
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
void InternalSharedPtr<T>::weak_reference_() noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
      assert(use_count());
    }
    ctrl_->weak_ref();
  }
}

template <typename T>
void InternalSharedPtr<T>::weak_dereference_() noexcept
{
  if (ctrl_) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(get());
    }
    ctrl_->weak_deref();
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
template <typename U, typename V>
void InternalSharedPtr<T>::init_shared_from_this_(const EnableSharedFromThis<U>* weak, V* ptr)
{
  if (weak && weak->weak_this_.expired()) {
    using RawU = std::remove_cv_t<U>;

    weak->weak_this_ =
      InternalSharedPtr<RawU>{*this, const_cast<RawU*>(static_cast<const U*>(ptr))};
  }
}

template <typename T>
void InternalSharedPtr<T>::init_shared_from_this_(...)
{
}

// Every non-trivial constructor goes through this function!
template <typename T>
template <typename U>
InternalSharedPtr<T>::InternalSharedPtr(AllocatedControlBlockTag,
                                        control_block_type* ctrl_impl,
                                        U* ptr) noexcept
  : ctrl_{ctrl_impl}, ptr_{ptr}
{
  init_shared_from_this_(ptr, ptr);
}

// ==========================================================================================

template <typename T>
InternalSharedPtr<T>::InternalSharedPtr(std::nullptr_t) noexcept
{
}

// clang-format off
template <typename T>
template <typename U, typename D, typename A, typename SFINAE>
 InternalSharedPtr<T>::InternalSharedPtr(U* ptr, D deleter, A allocator) try :
  InternalSharedPtr{
    AllocatedControlBlockTag{},
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
InternalSharedPtr<T>::InternalSharedPtr(element_type* ptr)
  : InternalSharedPtr{ptr, std::default_delete<element_type>{}}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(U* ptr) : InternalSharedPtr{ptr, std::default_delete<U>{}}
{
  static_assert(!std::is_void_v<U>, "incomplete type");
  // NOLINTNEXTLINE(bugprone-sizeof-expression)
  static_assert(sizeof(U) > 0, "incomplete type");
}

template <typename T>
InternalSharedPtr<T>::InternalSharedPtr(const InternalSharedPtr& other) noexcept
  : InternalSharedPtr{AllocatedControlBlockTag{}, other.ctrl_, other.ptr_}
{
  strong_reference_();
}

template <typename T>
InternalSharedPtr<T>&
InternalSharedPtr<T>::operator=(  // NOLINT(bugprone-unhandled-self-assignment): yes it does
  const InternalSharedPtr& other) noexcept
{
  InternalSharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
InternalSharedPtr<T>::InternalSharedPtr(InternalSharedPtr&& other) noexcept
  : InternalSharedPtr{AllocatedControlBlockTag{},
                      std::exchange(other.ctrl_, nullptr),
                      std::exchange(other.ptr_, nullptr)}
{
  // we do not increment ref-counts here, the other pointer gives up ownership entirely (so
  // refcounts stay at previous levels)
}

template <typename T>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(InternalSharedPtr&& other) noexcept
{
  InternalSharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(const InternalSharedPtr<U>& other) noexcept
  : InternalSharedPtr{AllocatedControlBlockTag{}, other.ctrl_, other.ptr_}
{
  strong_reference_();
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(const InternalSharedPtr<U>& other) noexcept
{
  InternalSharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(InternalSharedPtr<U>&& other) noexcept
  : InternalSharedPtr{AllocatedControlBlockTag{},
                      std::exchange(other.ctrl_, nullptr),
                      std::exchange(other.ptr_, nullptr)}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(InternalSharedPtr<U>&& other) noexcept
{
  InternalSharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename D, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(std::unique_ptr<U, D>&& ptr)
  : InternalSharedPtr{ptr.get(), std::move(ptr.get_deleter())}
{
  // Release only after we have fully constructed ourselves to preserve strong exception
  // guarantee. If the above constructor throws, this has no effect.
  static_cast<void>(ptr.release());
}

template <typename T>
template <typename U, typename D, typename SFINAE>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(std::unique_ptr<U, D>&& ptr)
{
  InternalSharedPtr{std::move(ptr)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(const SharedPtr<U>& other) noexcept
  : InternalSharedPtr{other.internal_ptr({})}
{
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(const SharedPtr<U>& other) noexcept
{
  InternalSharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(SharedPtr<U>&& other) noexcept
  : InternalSharedPtr{std::move(other.internal_ptr({}))}
{
  // Normally, move-assigning from one shared ptr instance to another does not incur any
  // reference-count updating, however in this case we are "down-casting" from a user reference
  // to just an internal reference. So we must decrement the user count since other is empty
  // after this call.
  user_dereference_({});
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>& InternalSharedPtr<T>::operator=(SharedPtr<U>&& other) noexcept
{
  InternalSharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
InternalSharedPtr<T>::InternalSharedPtr(const InternalWeakPtr<U>& other)
  : InternalSharedPtr{AllocatedControlBlockTag{},
                      other.ctrl_,
                      other.ctrl_ ? static_cast<T*>(other.ctrl_->ptr()) : nullptr}
{
  if (!ctrl_) {
    throw BadInternalWeakPtr{
      "Trying to construct an InternalSharedPtr from "
      "an empty InternalWeakPtr"};
  }
  strong_reference_();
}

template <typename T>
template <typename U>
InternalSharedPtr<T>::InternalSharedPtr(const InternalSharedPtr<U>& other,
                                        element_type* ptr) noexcept
  // Do not use the AllocatedControlBlockTag ctor, otherwise this will infinitely loop for
  // enable_shared_from_this classes!
  : ctrl_{other.ctrl_}, ptr_{ptr}
{
  strong_reference_();
}

template <typename T>
template <typename U>
InternalSharedPtr<T>::InternalSharedPtr(InternalSharedPtr<U>&& other, element_type* ptr) noexcept
  : InternalSharedPtr{AllocatedControlBlockTag{}, other.ctrl_, ptr}
{
  other.ctrl_ = nullptr;
  other.ptr_  = nullptr;
}

template <typename T>
InternalSharedPtr<T>::~InternalSharedPtr() noexcept
{
  strong_dereference_();
}

// ==========================================================================================

template <typename T>
void InternalSharedPtr<T>::swap(InternalSharedPtr& other) noexcept
{
  using std::swap;

  swap(other.ctrl_, ctrl_);
  swap(other.ptr_, ptr_);
}

template <typename T>
void swap(InternalSharedPtr<T>& lhs, InternalSharedPtr<T>& rhs) noexcept
{
  lhs.swap(rhs);
}

template <typename T>
void InternalSharedPtr<T>::reset() noexcept
{
  InternalSharedPtr<T>{}.swap(*this);
}

template <typename T>
void InternalSharedPtr<T>::reset(std::nullptr_t) noexcept
{
  reset();
}

template <typename T>
template <typename U, typename D, typename A, typename SFINAE>
void InternalSharedPtr<T>::reset(U* ptr, D deleter, A allocator)
{
  InternalSharedPtr<T>{ptr, std::move(deleter), std::move(allocator)}.swap(*this);
}

// ==========================================================================================

template <typename T>
typename InternalSharedPtr<T>::element_type* InternalSharedPtr<T>::get() const noexcept
{
  return ptr_;
}

template <typename T>
typename InternalSharedPtr<T>::element_type& InternalSharedPtr<T>::operator*() const noexcept
{
  return *ptr_;
}

template <typename T>
typename InternalSharedPtr<T>::element_type* InternalSharedPtr<T>::operator->() const noexcept
{
  return get();
}

template <typename T>
typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::use_count() const noexcept
{
  // SharedPtr's are a subset of InternalSharedPtr (since each one holds an InternalSharedPtr),
  // so the number of strong references gives the total unique references held to the pointer.
  return strong_ref_count();
}

template <typename T>
typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::strong_ref_count()
  const noexcept
{
  return ctrl_ ? ctrl_->strong_ref_cnt() : 0;
}

template <typename T>
typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::user_ref_count() const noexcept
{
  return ctrl_ ? ctrl_->user_ref_cnt() : 0;
}

template <typename T>
typename InternalSharedPtr<T>::ref_count_type InternalSharedPtr<T>::weak_ref_count() const noexcept
{
  return ctrl_ ? ctrl_->weak_ref_cnt() : 0;
}

template <typename T>
InternalSharedPtr<T>::operator bool() const noexcept
{
  return get() != nullptr;
}

template <typename T>
SharedPtr<T> InternalSharedPtr<T>::as_user_ptr() const noexcept
{
  return SharedPtr<T>{*this};
}

// ==========================================================================================

template <typename T, typename... Args>
InternalSharedPtr<T> make_internal_shared(Args&&... args)
{
  using RawT           = std::remove_cv_t<T>;
  using allocator_type = std::allocator<RawT>;

  auto alloc = allocator_type{};
  auto control_block =
    detail::construct_from_allocator_<detail::InplaceControlBlock<RawT, allocator_type>>(
      alloc, static_cast<RawT*>(nullptr), alloc, std::forward<Args>(args)...);
  return {typename InternalSharedPtr<T>::AllocatedControlBlockTag{},
          control_block,
          static_cast<RawT*>(control_block->ptr())};
}

// ==========================================================================================

template <typename T, typename U>
InternalSharedPtr<T> static_pointer_cast(const InternalSharedPtr<U>& ptr) noexcept
{
  return {ptr, static_cast<typename InternalSharedPtr<T>::element_type*>(ptr.get())};
}

// ==========================================================================================

template <typename T, typename U>
bool operator==(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() != rhs.get();
}

template <typename T, typename U>
bool operator<(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() < rhs.get();
}

template <typename T, typename U>
bool operator>(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() > rhs.get();
}

template <typename T, typename U>
bool operator<=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() <= rhs.get();
}

template <typename T, typename U>
bool operator>=(const InternalSharedPtr<T>& lhs, const InternalSharedPtr<U>& rhs) noexcept
{
  return lhs.get() >= rhs.get();
}

// ==========================================================================================

template <typename T>
bool operator==(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() == nullptr;
}

template <typename T>
bool operator==(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr == rhs.get();
}

template <typename T>
bool operator!=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() != nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr != rhs.get();
}

template <typename T>
bool operator<(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() < nullptr;
}

template <typename T>
bool operator<(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr < rhs.get();
}

template <typename T>
bool operator>(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() > nullptr;
}

template <typename T>
bool operator>(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr > rhs.get();
}

template <typename T>
bool operator<=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() <= nullptr;
}

template <typename T>
bool operator<=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr <= rhs.get();
}

template <typename T>
bool operator>=(const InternalSharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() >= nullptr;
}

template <typename T>
bool operator>=(std::nullptr_t, const InternalSharedPtr<T>& rhs) noexcept
{
  return nullptr >= rhs.get();
}

// ==========================================================================================

template <typename T>
constexpr EnableSharedFromThis<T>::EnableSharedFromThis(const EnableSharedFromThis&) noexcept
{
}

template <typename T>
constexpr EnableSharedFromThis<T>& EnableSharedFromThis<T>::operator=(
  const EnableSharedFromThis&) noexcept
{
  return *this;
}

// ==========================================================================================

template <typename T>
typename EnableSharedFromThis<T>::shared_type EnableSharedFromThis<T>::shared_from_this()
{
  return shared_type{weak_this_};
}

template <typename T>
typename EnableSharedFromThis<T>::const_shared_type EnableSharedFromThis<T>::shared_from_this()
  const
{
  return const_shared_type{weak_this_};
}

}  // namespace legate

namespace std {

template <typename T>
size_t hash<legate::InternalSharedPtr<T>>::operator()(
  const legate::InternalSharedPtr<T>& ptr) const noexcept
{
  return hash<typename legate::InternalSharedPtr<T>::element_type*>{}(ptr.get());
}

}  // namespace std
