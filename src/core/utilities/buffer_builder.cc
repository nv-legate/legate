/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "core/utilities/buffer_builder.h"

namespace legate {

BufferBuilder::BufferBuilder()
{
  // Reserve 4KB to minimize resizing while packing the arguments.
  buffer_.reserve(4096);
}

void BufferBuilder::pack_buffer(const void* src, size_t size)
{
  auto tgt = buffer_.data() + buffer_.size();
  buffer_.resize(buffer_.size() + size);
  memcpy(tgt, src, size);
}

Legion::UntypedBuffer BufferBuilder::to_legion_buffer() const
{
  return Legion::UntypedBuffer(buffer_.data(), buffer_.size());
}

}  // namespace legate
