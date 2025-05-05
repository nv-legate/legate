/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/buffer.h>

#include <legate/data/buffer.h>
#include <legate/data/inline_allocation.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/dispatch.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate::detail {

namespace {

class GetInlineAllocationPtr {
 public:
  template <Type::Code CODE, std::int32_t DIM>
  [[nodiscard]] void* operator()(const Legion::UntypedDeferredBuffer<>& buf,
                                 std::size_t* strides_data) const
  {
    const auto typed_buf = static_cast<Buffer<type_of_t<CODE>, DIM>>(buf);

    return typed_buf.ptr(typed_buf.get_bounds(), strides_data);
  }
};

}  // namespace

TaskLocalBuffer::TaskLocalBuffer(const Legion::UntypedDeferredBuffer<>& buf,
                                 InternalSharedPtr<Type> type,
                                 // Domain is a POD type, no point in moving it
                                 const Domain& bounds)  // NOLINT(modernize-pass-by-value)
  : buf_{buf}, type_{std::move(type)}, domain_{bounds}
{
}

mapping::StoreTarget TaskLocalBuffer::memory_kind() const
{
  return mapping::detail::to_target(legion_buffer().get_instance().get_location().kind());
}

InlineAllocation TaskLocalBuffer::get_inline_allocation() const
{
  auto strides = std::vector<std::size_t>(static_cast<std::size_t>(dim()), 0);
  auto* const ptr =
    double_dispatch(dim(), type()->code, GetInlineAllocationPtr{}, legion_buffer(), strides.data());
  const auto type_size = type()->size();

  // The strides are supposed to be in *bytes*, not number of elements, so we need to convert
  // to that before we pass this off to numpy. We cannot initialize the vector with it,
  // because Legion overwrites the vector wholesale.
  for (auto& s : strides) {
    s *= type_size;
  }
  return {ptr, std::move(strides), memory_kind()};
}

}  // namespace legate::detail
