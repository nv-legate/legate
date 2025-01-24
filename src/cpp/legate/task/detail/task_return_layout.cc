/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/task/detail/task_return_layout.h>

#include <legate/utilities/detail/deserializer.h>

#include <limits>
#include <memory>

namespace legate::detail {

std::size_t TaskReturnLayoutForUnpack::next(std::size_t element_size, std::size_t alignment)
{
  // Need to shift the pointer a bit so we don't pass a null pointer to the first alignment call.
  // Suppress the clang tidy warning as we never access this fake pointer anyway
  auto ptr = reinterpret_cast<void*>(current_offset_ + BASE);  // NOLINT(performance-no-int-to-ptr)
  // We can assume the buffer has an "infinite" size, as we will create a buffer big enough based on
  // this offset calculation
  auto capacity = std::numeric_limits<std::size_t>::max();

  auto&& [next_ptr, _] = align_for_unpack<std::int8_t>(ptr, capacity, element_size, alignment);

  // Shift the offset back so we can return it
  auto result = reinterpret_cast<std::size_t>(next_ptr) - BASE;

  // Increment the current offset by the size of the value to be written
  current_offset_ = result + element_size;

  return result;
}

TaskReturnLayoutForPack::TaskReturnLayoutForPack(const std::vector<ReturnValue>& return_values)
  : TaskReturnLayoutForUnpack{0}
{
  offsets_.reserve(return_values.size());

  for (auto&& ret : return_values) {
    offsets_.push_back(next(ret.size(), ret.alignment()));
  }
}

}  // namespace legate::detail
