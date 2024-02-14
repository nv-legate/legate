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

#include "core/data/allocator.h"

#include "core/data/buffer.h"
#include "core/utilities/typedefs.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace legate {

class ScopedAllocator::Impl {
 public:
  using ByteBuffer = Buffer<std::int8_t>;

  Impl(Memory::Kind kind, bool scoped, std::size_t alignment);
  ~Impl() noexcept;

  [[nodiscard]] void* allocate(std::size_t bytes);
  void deallocate(void* ptr);

 private:
  Memory::Kind target_kind_{Memory::Kind::SYSTEM_MEM};
  bool scoped_{};
  std::size_t alignment_{};
  std::unordered_map<const void*, ByteBuffer> buffers_{};
};

ScopedAllocator::Impl::Impl(Memory::Kind kind, bool scoped, std::size_t alignment)
  : target_kind_{kind}, scoped_{scoped}, alignment_{alignment}
{
}

ScopedAllocator::Impl::~Impl() noexcept
{
  if (scoped_) {
    for (auto& pair : buffers_) {
      pair.second.destroy();
    }
    buffers_.clear();
  }
}

void* ScopedAllocator::Impl::allocate(std::size_t bytes)
{
  if (bytes == 0) {
    return nullptr;
  }

  auto buffer = create_buffer<std::int8_t>(bytes, target_kind_, alignment_);
  auto ptr    = buffer.ptr(0);

  buffers_[ptr] = std::move(buffer);
  return ptr;
}

void ScopedAllocator::Impl::deallocate(void* ptr)
{
  auto finder = buffers_.find(ptr);
  if (finder == buffers_.end()) {
    throw std::runtime_error{"Invalid address for deallocation"};
  }

  auto buffer = finder->second;
  buffers_.erase(finder);
  buffer.destroy();
}

ScopedAllocator::ScopedAllocator(Memory::Kind kind, bool scoped, std::size_t alignment)
  : impl_{new Impl{kind, scoped, alignment}}
{
}

void* ScopedAllocator::allocate(std::size_t bytes) { return impl_->allocate(bytes); }

void ScopedAllocator::deallocate(void* ptr) { impl_->deallocate(ptr); }

void ScopedAllocator::ImplDeleter::operator()(Impl* ptr) const noexcept { delete ptr; }

}  // namespace legate
