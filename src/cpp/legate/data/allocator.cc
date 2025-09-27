/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/allocator.h>

#include <legate/data/buffer.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace legate {

/**
 * @brief Detail-hiding implementation class for `ScopedAllocator`.
 *
 * @see ScopedAllocator
 */
class ScopedAllocator::Impl {
 public:
  using ByteBuffer = Buffer<std::int8_t>;

  // @brief See explicit constructor of `ScopedAllocator`
  Impl(Memory::Kind kind, bool scoped, std::size_t alignment);

  ~Impl() noexcept;

  // @brief See `ScopedAllocator::allocate()`
  [[nodiscard]] void* allocate(std::size_t bytes);

  // @brief See `ScopedAllocator::allocate_aligned()`
  [[nodiscard]] void* allocate_aligned(std::size_t bytes, std::size_t alignment);

  // @brief See `ScopedAllocator::deallocate()`
  void deallocate(void* ptr);

 private:
  Memory::Kind target_kind_{Memory::Kind::SYSTEM_MEM};
  bool scoped_{};
  std::size_t alignment_{};
  std::unordered_map<const void*, ByteBuffer> buffers_{};
};

namespace {

// @brief Whether `n` is a power of 2
constexpr bool is_power_of_2(std::size_t n) { return n != 0 && ((n & (n - 1)) == 0); }

}  // namespace

ScopedAllocator::Impl::Impl(Memory::Kind kind, bool scoped, std::size_t alignment)
  : target_kind_{kind}, scoped_{scoped}, alignment_{alignment}
{
  if (!is_power_of_2(alignment)) {
    throw detail::TracedException<std::domain_error>{
      fmt::format("invalid alignment {}, must be a power of 2", alignment)};
  }
}

ScopedAllocator::Impl::~Impl() noexcept
{
  if (scoped_) {
    for (auto&& pair : buffers_) {
      pair.second.destroy();
    }
    buffers_.clear();
  }
}

void* ScopedAllocator::Impl::allocate_aligned(std::size_t bytes, std::size_t alignment)
{
  if (!is_power_of_2(alignment)) {
    throw detail::TracedException<std::domain_error>{
      fmt::format("invalid alignment {}, must be a power of 2", alignment)};
  }
  if (bytes == 0) {
    return nullptr;
  }

  auto buffer = create_buffer<std::int8_t>(bytes, target_kind_, std::max(alignment_, alignment));
  auto* ptr   = buffer.ptr(0);

  try {
    buffers_[ptr] = std::move(buffer);
  } catch (...) {
    buffer.destroy();
    throw;
  }
  return ptr;
}

void* ScopedAllocator::Impl::allocate(std::size_t bytes)
{
  return allocate_aligned(bytes, alignment_);
}

void ScopedAllocator::Impl::deallocate(void* ptr)
{
  if (!ptr) {
    return;
  }

  const auto it = buffers_.find(ptr);

  if (it == buffers_.end()) {
    throw detail::TracedException<std::invalid_argument>{
      fmt::format("Invalid address {} for deallocation", ptr)};
  }

  it->second.destroy();
  buffers_.erase(it);
}

ScopedAllocator::ScopedAllocator(Memory::Kind kind, bool scoped, std::size_t alignment)
  : impl_{std::make_shared<Impl>(kind, scoped, alignment)}
{
}

void* ScopedAllocator::allocate(std::size_t bytes) { return impl_->allocate(bytes); }

void* ScopedAllocator::allocate_aligned(std::size_t bytes, std::size_t alignment)
{
  return impl_->allocate_aligned(bytes, alignment);
}

void ScopedAllocator::deallocate(void* ptr) { impl_->deallocate(ptr); }

ScopedAllocator::~ScopedAllocator() noexcept = default;

}  // namespace legate
