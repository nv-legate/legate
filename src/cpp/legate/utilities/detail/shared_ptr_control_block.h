/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/compressed_pair.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace legate::detail {

class ControlBlockBase {
 public:
  using ref_count_type = std::uint32_t;

  constexpr ControlBlockBase() noexcept = default;
  virtual ~ControlBlockBase() noexcept  = default;

  // To prevent object slicing
  ControlBlockBase(const ControlBlockBase&)            = delete;
  ControlBlockBase& operator=(const ControlBlockBase&) = delete;

  virtual void destroy_object() noexcept                 = 0;
  virtual void destroy_control_block() noexcept          = 0;
  [[nodiscard]] virtual void* ptr() noexcept             = 0;
  [[nodiscard]] virtual const void* ptr() const noexcept = 0;

  void maybe_destroy_control_block() noexcept;

  [[nodiscard]] ref_count_type strong_ref_cnt() const noexcept;
  [[nodiscard]] ref_count_type weak_ref_cnt() const noexcept;
  [[nodiscard]] ref_count_type user_ref_cnt() const noexcept;

  ref_count_type strong_ref() noexcept;
  ref_count_type weak_ref() noexcept;
  ref_count_type user_ref() noexcept;
  ref_count_type strong_deref() noexcept;
  ref_count_type weak_deref() noexcept;
  ref_count_type user_deref() noexcept;

 protected:
  template <typename T>
  static void destroy_control_block_impl_(T* cb_impl) noexcept;

 private:
  [[nodiscard]] static ref_count_type load_refcount_(
    const std::atomic<ref_count_type>& refcount) noexcept;
  [[nodiscard]] static ref_count_type increment_refcount_(
    std::atomic<ref_count_type>* refcount) noexcept;
  [[nodiscard]] static ref_count_type decrement_refcount_(
    std::atomic<ref_count_type>* refcount) noexcept;

  std::atomic<ref_count_type> strong_refs_{1};  // The number of InternalSharedPtr's
  std::atomic<ref_count_type> weak_refs_{};     // The number of InternalWeakPtr
  std::atomic<ref_count_type> user_refs_{};     // The number of SharedPtr
};

// ==========================================================================================

template <typename T, typename Deleter, typename Alloc>
class SeparateControlBlock final : public ControlBlockBase {
 public:
  using value_type     = T;
  using deleter_type   = Deleter;
  using allocator_type = Alloc;

  SeparateControlBlock() = delete;

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  SeparateControlBlock(value_type* ptr, deleter_type deleter, allocator_type allocator) noexcept;

  void destroy_object() noexcept override;

  void destroy_control_block() noexcept override;

  [[nodiscard]] void* ptr() noexcept override;
  [[nodiscard]] const void* ptr() const noexcept override;

  template <typename U>
  [[nodiscard]] auto rebind_alloc() const;

 private:
  [[nodiscard]] allocator_type& alloc_() noexcept;
  [[nodiscard]] const allocator_type& alloc_() const noexcept;

  [[nodiscard]] deleter_type& deleter_() noexcept;
  [[nodiscard]] const deleter_type& deleter_() const noexcept;

  value_type* ptr_;
  CompressedPair<deleter_type, allocator_type> pair_;
};

// ==========================================================================================

template <typename T, typename Allocator>
class InplaceControlBlock final : public ControlBlockBase {
 public:
  using value_type     = T;
  using allocator_type = Allocator;

 private:
  class AlignedStorage {
   public:
    constexpr AlignedStorage() noexcept = default;
    // use this ctor to avoid zero-initializing the array
    // NOLINTNEXTLINE(google-explicit-constructor) to mimic std::pair constructor
    constexpr AlignedStorage(std::nullptr_t) noexcept {}

    [[nodiscard]] void* addr() noexcept { return static_cast<void*>(&mem); }
    [[nodiscard]] const void* addr() const noexcept { return static_cast<const void*>(&mem); }

    alignas(alignof(value_type)) std::byte mem[sizeof(value_type)];
  };

 public:
  InplaceControlBlock() = delete;

  template <typename... Args>
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  explicit InplaceControlBlock(allocator_type allocator, Args&&... args);

  void destroy_object() noexcept override;

  void destroy_control_block() noexcept override;

  [[nodiscard]] void* ptr() noexcept override;
  [[nodiscard]] const void* ptr() const noexcept override;

  template <typename U>
  [[nodiscard]] auto rebind_alloc() const;

 private:
  [[nodiscard]] allocator_type& alloc_() noexcept;
  [[nodiscard]] const allocator_type& alloc_() const noexcept;

  [[nodiscard]] AlignedStorage& store_() noexcept;
  [[nodiscard]] const AlignedStorage& store_() const noexcept;

  CompressedPair<allocator_type, AlignedStorage> pair_;
};

// ==========================================================================================

template <typename U, typename Alloc, typename P, typename... Args>
[[nodiscard]] U* construct_from_allocator_(  // NOLINT(readability-identifier-naming)
  Alloc& allocator,
  P* hint,
  Args&&... args);

}  // namespace legate::detail

#include <legate/utilities/detail/shared_ptr_control_block.inl>
