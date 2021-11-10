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

namespace legate {

template <typename T>
void BufferBuilder::pack(const T& value)
{
  pack_buffer(reinterpret_cast<const int8_t*>(&value), sizeof(T));
}

template <typename T>
void BufferBuilder::pack(const std::vector<T>& values)
{
  uint32_t size = values.size();
  pack(size);
  pack_buffer(values.data(), size * sizeof(T));
}

}  // namespace legate
