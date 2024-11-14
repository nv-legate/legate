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

#include "legate/data/detail/future_wrapper.h"

#include "legate/mapping/detail/mapping.h"
#include "legate/runtime/detail/runtime.h"
#include "legate/utilities/dispatch.h"
#include "legate/utilities/machine.h"
#include "legate/utilities/macros.h"

#include <cstring>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

[[nodiscard]] Legion::UntypedDeferredValue untyped_deferred_value_from_future(
  const Legion::Future& fut, std::size_t field_size, std::size_t field_offset)
{
  const auto mem_kind =
    find_memory_kind_for_executing_processor(LEGATE_DEFINED(LEGATE_NO_FUTURES_ON_FB));

  if (!fut.valid()) {
    return Legion::UntypedDeferredValue{field_size, mem_kind};
  }

  LEGATE_ASSERT(field_offset + field_size <= fut.get_untyped_size());

  const auto* init_value = static_cast<const std::int8_t*>(fut.get_buffer(mem_kind)) + field_offset;

  if (LEGATE_DEFINED(LEGATE_USE_CUDA) && (mem_kind == Memory::Kind::GPU_FB_MEM)) {
    // TODO(wonchanl): This should be done by Legion
    auto ret       = Legion::UntypedDeferredValue{field_size, mem_kind};
    const auto acc = AccessorWO<std::int8_t, 1>{ret, field_size, false};
    auto* runtime  = Runtime::get_runtime();
    auto stream    = runtime->get_cuda_stream();

    runtime->get_cuda_driver_api()->mem_cpy_async(acc.ptr(0), init_value, field_size, stream);
    return ret;
  }
  return Legion::UntypedDeferredValue{field_size, mem_kind, init_value};
}

}  // namespace

// Silence pass-by-value since Legion::Domain is POD, and the move ctor just does the copy
// anyways. Unfortunately there is no way to check this programatically (e.g. via a
// static_assert).
FutureWrapper::FutureWrapper(bool read_only,
                             std::uint32_t field_size,
                             std::size_t field_offset,
                             const Domain& domain,  // NOLINT(modernize-pass-by-value)
                             Legion::Future future)
  : read_only_{read_only},
    field_size_{field_size},
    field_offset_{field_offset},
    domain_{domain},
    future_{std::move(future)},
    buffer_{read_only ? Legion::UntypedDeferredValue{}
                      : untyped_deferred_value_from_future(
                          get_future(), this->field_size(), this->field_offset())}
{
}

FutureWrapper::~FutureWrapper() noexcept
{
  if (has_started() || !future_.exists()) {
    return;
  }
  // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
  // this would be collected by the Legion runtime
  static_cast<void>(std::make_unique<Legion::Future>(std::move(future_)).release());
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

namespace {

class GetInlineAllocFromFuture {
  template <std::int32_t DIM>
  [[nodiscard]] static Rect<DIM> domain_to_rect_(const Domain& domain)
  {
    return domain.dim > 0 ? Rect<DIM>{domain}
                          : Rect<DIM>{Point<DIM>::ZEROES(), Point<DIM>::ZEROES()};
  }

 public:
  template <std::int32_t DIM>
  [[nodiscard]] InlineAllocation operator()(const Legion::Future& future,
                                            const Domain& domain,
                                            std::size_t field_size,
                                            std::size_t field_offset)
  {
    const auto rect = domain_to_rect_<DIM>(domain);
    const auto acc  = AccessorRO<std::int8_t, DIM>{future,
                                                   rect,
                                                   Memory::Kind::NO_MEMKIND,
                                                   field_size,
                                                   false /*check_field_size*/,
                                                   false,
                                                   nullptr,
                                                   field_offset};

    auto strides = std::vector<std::size_t>(DIM, 0);
    auto ptr     = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));

    return {ptr, std::move(strides)};
  }

  template <std::int32_t DIM>
  [[nodiscard]] InlineAllocation operator()(const Legion::UntypedDeferredValue& value,
                                            const Domain& domain,
                                            std::size_t field_size)
  {
    const auto rect = domain_to_rect_<DIM>(domain);
    const auto acc =
      AccessorRO<std::int8_t, DIM>{value, rect, field_size, false /*check_field_size*/};

    auto strides = std::vector<std::size_t>(DIM, 0);
    auto ptr     = const_cast<void*>(static_cast<const void*>(acc.ptr(rect, strides.data())));

    return {ptr, std::move(strides)};
  }
};

}  // namespace

InlineAllocation FutureWrapper::get_inline_allocation(const Domain& domain) const
{
  if (is_read_only()) {
    return dim_dispatch(std::max(1, domain.dim),
                        GetInlineAllocFromFuture{},
                        get_future(),
                        domain,
                        field_size(),
                        field_offset());
  }
  return dim_dispatch(
    std::max(1, domain.dim), GetInlineAllocFromFuture{}, get_buffer(), domain, field_size());
}

InlineAllocation FutureWrapper::get_inline_allocation() const
{
  return get_inline_allocation(domain());
}

mapping::StoreTarget FutureWrapper::target() const
{
  // TODO(wonchanl): The following is not entirely accurate, as the custom mapper can override the
  // default mapping policy for futures. Unfortunately, Legion doesn't expose mapping decisions
  // for futures, but instead would move the data wherever it's requested. Until Legate gets
  // access to that information, we potentially give inaccurate answers
  return mapping::detail::to_target(
    find_memory_kind_for_executing_processor(LEGATE_DEFINED(LEGATE_NO_FUTURES_ON_FB)));
}

// Initializing isn't a const operation, even if all member functions are used const-ly
// NOLINTNEXTLINE(readability-make-member-function-const)
void FutureWrapper::initialize_with_identity(GlobalRedopID redop_id)
{
  const auto untyped_acc = AccessorWO<std::int8_t, 1>{get_buffer(), field_size()};
  auto* ptr              = untyped_acc.ptr(0);
  const auto* redop =
    Legion::Runtime::get_reduction_op(static_cast<Legion::ReductionOpID>(redop_id));
  const auto* identity = redop->identity;

  LEGATE_ASSERT(redop->sizeof_lhs == field_size());
  if (LEGATE_DEFINED(LEGATE_USE_CUDA) &&
      (get_buffer().get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM)) {
    auto* runtime = Runtime::get_runtime();
    auto stream   = runtime->get_cuda_stream();

    runtime->get_cuda_driver_api()->mem_cpy_async(ptr, identity, field_size(), stream);
  } else {
    std::memcpy(ptr, identity, field_size());
  }
}

ReturnValue FutureWrapper::pack(const InternalSharedPtr<Type>& type) const
{
  return {get_buffer(), field_size(), type->alignment()};
}

const void* FutureWrapper::get_untyped_pointer_from_future() const
{
  LEGATE_ASSERT(get_future().valid());
  LEGATE_ASSERT(field_offset() + field_size() <= get_future().get_untyped_size());
  return static_cast<const std::int8_t*>(
           get_future().get_buffer(find_memory_kind_for_executing_processor())) +
         field_offset();
}

}  // namespace legate::detail
