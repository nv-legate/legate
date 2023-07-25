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

namespace legate {

class ScopedAllocator::Impl {
 public:
  using ByteBuffer = Buffer<int8_t>;

 public:
  Impl(Memory::Kind kind, bool scoped, size_t alignment);
  ~Impl();

 public:
  void* allocate(size_t bytes);
  void deallocate(void* ptr);

 private:
  Memory::Kind target_kind_{Memory::Kind::SYSTEM_MEM};
  bool scoped_;
  size_t alignment_;
  std::unordered_map<const void*, ByteBuffer> buffers_{};
};

ScopedAllocator::Impl::Impl(Memory::Kind kind, bool scoped, size_t alignment)
  : target_kind_(kind), scoped_(scoped), alignment_(alignment)
{
}

ScopedAllocator::Impl::~Impl()
{
  if (scoped_) {
    for (auto& pair : buffers_) { pair.second.destroy(); }
    buffers_.clear();
  }
}

void* ScopedAllocator::Impl::allocate(size_t bytes)
{
  if (bytes == 0) return nullptr;

  ByteBuffer buffer = create_buffer<int8_t>(bytes, target_kind_, alignment_);

  void* ptr = buffer.ptr(0);

  buffers_[ptr] = buffer;
  return ptr;
}

void ScopedAllocator::Impl::deallocate(void* ptr)
{
  ByteBuffer buffer;
  auto finder = buffers_.find(ptr);
  if (finder == buffers_.end()) { throw std::runtime_error("Invalid address for deallocation"); }

  buffer = finder->second;
  buffers_.erase(finder);
  buffer.destroy();
}

ScopedAllocator::ScopedAllocator(Memory::Kind kind, bool scoped, size_t alignment)
  : impl_(new Impl(kind, scoped, alignment))
{
}

ScopedAllocator::~ScopedAllocator() { delete impl_; }

void* ScopedAllocator::allocate(size_t bytes) { return impl_->allocate(bytes); }

void ScopedAllocator::deallocate(void* ptr) { impl_->deallocate(ptr); }

ScopedAllocator::ScopedAllocator(ScopedAllocator&& other) : impl_(other.impl_)
{
  other.impl_ = nullptr;
}

ScopedAllocator& ScopedAllocator::operator=(ScopedAllocator&& other)
{
  impl_       = other.impl_;
  other.impl_ = nullptr;
  return *this;
}

}  // namespace legate
