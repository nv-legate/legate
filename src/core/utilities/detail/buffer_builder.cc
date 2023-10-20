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

#include "core/utilities/detail/buffer_builder.h"

#include "legate_defines.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace legate::detail {

BufferBuilder::BufferBuilder()
{
  // Reserve 4KB to minimize resizing while packing the arguments.
  buffer_.reserve(4096);
}

void BufferBuilder::pack_buffer(const void* mem, std::size_t size, std::size_t align)
{
  if (!size) return;

  if (LegateDefined(LEGATE_USE_DEBUG)) {
    constexpr auto is_power_of_2 = [](std::size_t v) { return v && !(v & (v - 1)); };

    if (!align) throw std::invalid_argument{"alignment cannot be 0"};
    if (!is_power_of_2(align)) {
      throw std::invalid_argument{"alignment is not a power of 2: " + std::to_string(align)};
    }
  }
  if (!size) return;

  const auto orig_buf_size = buffer_.size();
  void* aligned_ptr        = nullptr;

  buffer_.resize(orig_buf_size + size);
  // do this after, resize() might reallocate the pointer
  aligned_ptr = buffer_.data() + orig_buf_size;
  if (!std::align(align, size, aligned_ptr, size)) {
    const auto orig_size_and_padding = size + align - 1;
    auto new_size_and_padding        = orig_size_and_padding;

    // the buffer was not exactly aligned, need to add on additional padding (which is at
    // most align - 1)
    buffer_.resize(orig_buf_size + orig_size_and_padding);
    // again, must reseat aligned_ptr in case resize() reallocated
    aligned_ptr    = buffer_.data() + orig_buf_size;
    const auto ptr = std::align(align, size, aligned_ptr, new_size_and_padding);
    // this should never fail, but hey you never know
    if (LegateDefined(LEGATE_USE_DEBUG) && !ptr) {
      std::stringstream ss;

      ss << "Failed to align pointer of size " << size << " to alignment " << align
         << ", this should never happen!";
      throw std::runtime_error{std::move(ss).str()};
    }
    // size + align - 1 is a potential overallocation. We must chop off the unneeded space at
    // the end since the next call expects buffer_.data() + buffer_.size() to be the beginning
    // of unused space.
    //
    // Since this is down-sizing, it is guaranteed not to reallocate.
    //
    //                             number of bytes that aligned_ptr was bumped up by
    //                             vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    buffer_.resize(orig_buf_size + (orig_size_and_padding - new_size_and_padding) + size);
  }
  std::memcpy(aligned_ptr, mem, size);
}

Legion::UntypedBuffer BufferBuilder::to_legion_buffer() const
{
  return Legion::UntypedBuffer(buffer_.data(), buffer_.size());
}

}  // namespace legate::detail
