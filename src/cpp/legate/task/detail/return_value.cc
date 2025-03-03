/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/return_value.h>

#include <cstring>

namespace legate::detail {

ReturnValue::ReturnValue(Legion::UntypedDeferredValue value,
                         std::size_t size,
                         std::size_t alignment)
  : value_{std::move(value)},
    size_{size},
    alignment_{alignment},
    is_device_value_{value_.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM}
{
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
