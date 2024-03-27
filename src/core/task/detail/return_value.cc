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

#include "core/task/detail/return_value.h"

#include <cstring>

namespace legate::detail {

ReturnValue::ReturnValue(Legion::UntypedDeferredValue value, std::size_t size)
  : value_{std::move(value)},
    size_{size},
    is_device_value_{value_.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM}
{
}

/*static*/ ReturnValue ReturnValue::unpack(const void* ptr,
                                           std::size_t size,
                                           Memory::Kind memory_kind)
{
  // We do not want to make this const since we want NRVO to kick in. If NRVO is not able to be
  // performed (for whatever reason), we want the value to be moved, and hence we cannot use
  // const.
  // NOLINTNEXTLINE(misc-const-correctness)
  ReturnValue result{Legion::UntypedDeferredValue{size, memory_kind}, size};

  LegateAssert(!result.is_device_value());

  const AccessorWO<std::int8_t, 1> acc{result.value_, result.size_, false};

  std::memcpy(acc.ptr(0), ptr, size);
  return result;
}

void ReturnValue::finalize(Legion::Context legion_context) const
{
  value_.finalize(legion_context);
}

void* ReturnValue::ptr()
{
  const AccessorRW<std::int8_t, 1> acc{value_, size_, false};
  return acc.ptr(0);
}

const void* ReturnValue::ptr() const
{
  const AccessorRO<std::int8_t, 1> acc{value_, size_, false};
  return acc.ptr(0);
}

}  // namespace legate::detail
