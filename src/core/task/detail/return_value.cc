/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
