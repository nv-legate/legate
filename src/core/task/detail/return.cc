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

#include "core/task/detail/return.h"

#include "core/runtime/detail/library.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"
#if LegateDefined(LEGATE_USE_CUDA)
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace legate::detail {

ReturnValue::ReturnValue(Legion::UntypedDeferredValue value, size_t size)
  : value_{std::move(value)},
    size_{size},
    is_device_value_{value_.get_instance().get_location().kind() == Memory::Kind::GPU_FB_MEM}
{
}

/*static*/ ReturnValue ReturnValue::unpack(const void* ptr, size_t size, Memory::Kind memory_kind)
{
  ReturnValue result{Legion::UntypedDeferredValue{size, memory_kind}, size};
  if (LegateDefined(LEGATE_USE_DEBUG)) {
    assert(!result.is_device_value());
  }
  memcpy(result.ptr(), ptr, size);

  return result;
}

void ReturnValue::finalize(Legion::Context legion_context) const
{
  value_.finalize(legion_context);
}

void* ReturnValue::ptr()
{
  const AccessorRW<int8_t, 1> acc{value_, size_, false};
  return acc.ptr(0);
}

const void* ReturnValue::ptr() const
{
  const AccessorRO<int8_t, 1> acc{value_, size_, false};
  return acc.ptr(0);
}

ReturnedException::ReturnedException(int32_t index, std::string_view error_message)
  : raised_{true}, index_{index}
{
  message_size_ =
    static_cast<decltype(message_size_)>(std::min(error_message_.size() - 1, error_message.size()));
  std::memcpy(error_message_.data(), error_message.data(), message_size_);
  if (message_size_ < error_message_.size()) {
    std::memset(error_message_.data() + message_size_, 0, error_message_.size() - message_size_);
  } else {
    error_message_.back() = '\0';  // always null-terminate
  }
}

std::optional<TaskException> ReturnedException::to_task_exception() const
{
  if (!raised_) {
    return std::nullopt;
  }
  return std::make_optional<TaskException>(
    {index_, {error_message_.data(), error_message_.data() + message_size_}});
}

namespace {

template <typename T>
constexpr size_t max_aligned_size_for_type()
{
  // NOLINTNEXTLINE(bugprone-sizeof-expression) // comparison to 0 is the point
  static_assert(sizeof(T) > 0, "Cannot be used for incomplete type");
  return sizeof(T) + alignof(T) - 1;
}

}  // namespace

// Note, this function returns an upper bound on the size of the type as it also incorporates
// alignment requirements for each member. It cannot know how much of the extra alignment
// padding it needs because that depends on how the input pointer is aligned when it goes to
// pack.
size_t ReturnedException::legion_buffer_size() const
{
  auto size = max_aligned_size_for_type<decltype(raised_)>();

  if (raised_) {
    size += max_aligned_size_for_type<decltype(index_)>();
    size += max_aligned_size_for_type<decltype(message_size_)>();
    size += message_size_;
  }
  return size;
}

namespace {

// REVIEW: why no BufferBuilder? Then we don't have to repeat this logic
template <typename T>
[[nodiscard]] std::pair<void*, size_t> pack_buffer(void* buf, size_t remaining_cap, T&& value)
{
  const auto [ptr, align_offset] = detail::align_for_unpack<T>(buf, remaining_cap);

  *static_cast<std::decay_t<T>*>(ptr) = std::forward<T>(value);
  return {static_cast<char*>(ptr) + sizeof(T), remaining_cap - sizeof(T) - align_offset};
}

}  // namespace

void ReturnedException::legion_serialize(void* buffer) const
{
  auto rem_cap = legion_buffer_size();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, raised_);
  if (raised_) {
    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, index_);
    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, message_size_);
    std::memcpy(buffer, error_message_.data(), message_size_);
  }
}

namespace {

template <typename T>
[[nodiscard]] std::pair<void*, size_t> unpack_buffer(void* buf, size_t remaining_cap, T* value)
{
  const auto [ptr, align_offset] = detail::align_for_unpack<T>(buf, remaining_cap);

  *value = *static_cast<std::decay_t<T>*>(ptr);
  return {static_cast<char*>(ptr) + sizeof(T), remaining_cap - sizeof(T) - align_offset};
}

}  // namespace

void ReturnedException::legion_deserialize(const void* buffer)
{
  // There is no information about the size of the buffer, nor can we know how much we need
  // until we pack all of it. So we just lie and say we have infinite memory.
  auto rem_cap = std::numeric_limits<size_t>::max();
  auto ptr     = const_cast<void*>(buffer);

  std::tie(ptr, rem_cap) = unpack_buffer(ptr, rem_cap, &raised_);
  if (raised_) {
    std::tie(ptr, rem_cap) = unpack_buffer(ptr, rem_cap, &index_);
    std::tie(ptr, rem_cap) = unpack_buffer(ptr, rem_cap, &message_size_);
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(message_size_ <= error_message_.size());
    }
    std::memcpy(error_message_.data(), ptr, message_size_);
    std::memset(error_message_.data() + message_size_, 0, error_message_.size() - message_size_);
    error_message_.back() = '\0';  // always null-terminate, regardless of size
  }
}

ReturnValue ReturnedException::pack() const
{
  auto buffer_size = legion_buffer_size();
  auto mem_kind    = find_memory_kind_for_executing_processor();
  auto buffer      = Legion::UntypedDeferredValue(buffer_size, mem_kind);

  const AccessorWO<int8_t, 1> acc{buffer, buffer_size, false};
  legion_serialize(acc.ptr(0));

  return {buffer, buffer_size};
}

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values)
  : return_values_(std::move(return_values))
{
  if (return_values_.size() > 1) {
    buffer_size_ += sizeof(uint32_t);
    for (auto& ret : return_values_) {
      buffer_size_ += sizeof(uint32_t) + ret.size();
    }
  } else if (!return_values_.empty()) {
    buffer_size_ = return_values_[0].size();
  }
}

void ReturnValues::legion_serialize(void* buffer) const
{
  // We pack N return values into the buffer in the following format:
  //
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  // |   #    | offset to |     | offset to  | total | value | value | ...
  // | values | scalar 1  | ... | scalar N-1 | value |   1   |   2   |
  // |        |           |     |            | size  |       |       |
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  //           <============ offsets ===============> <==== values =======>
  //
  // the size of value i is computed by offsets[i] - (i == 0 ? 0 : offsets[i-1])

  // Special case with a single scalar
  if (return_values_.size() == 1) {
    auto& ret = return_values_.front();

    if (ret.is_device_value()) {
      if (LegateDefined(LEGATE_USE_DEBUG)) {
        assert(Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC);
      }
      // TODO (jfaibussowit): expose cudaMemcpyAsync() as a stub instead
#if LegateDefined(LEGATE_USE_CUDA)
      CHECK_CUDA(cudaMemcpyAsync(buffer,
                                 ret.ptr(),
                                 ret.size(),
                                 cudaMemcpyDeviceToHost,
                                 cuda::StreamPool::get_stream_pool().get_stream()));
#endif
      return;
    }
    memcpy(buffer, ret.ptr(), ret.size());
    return;
  }

  *static_cast<uint32_t*>(buffer) = return_values_.size();
  auto ptr                        = static_cast<int8_t*>(buffer) + sizeof(uint32_t);

  uint32_t offset = 0;
  for (auto ret : return_values_) {
    offset += ret.size();
    *reinterpret_cast<uint32_t*>(ptr) = offset;
    ptr                               = ptr + sizeof(uint32_t);
  }

#if LegateDefined(LEGATE_USE_CUDA)
  if (Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();
    for (auto&& ret : return_values_) {
      const auto size = ret.size();
      if (ret.is_device_value()) {
        CHECK_CUDA(cudaMemcpyAsync(ptr, ret.ptr(), size, cudaMemcpyDeviceToHost, stream));
      } else {
        memcpy(ptr, ret.ptr(), size);
      }
      ptr += size;
    }
  } else
#endif
  {
    for (auto&& ret : return_values_) {
      const auto size = ret.size();
      memcpy(ptr, ret.ptr(), size);
      ptr += size;
    }
  }
}

void ReturnValues::legion_deserialize(const void* buffer)
{
  auto mem_kind = find_memory_kind_for_executing_processor();

  auto ptr        = static_cast<const int8_t*>(buffer);
  auto num_values = *reinterpret_cast<const uint32_t*>(ptr);

  auto offsets = reinterpret_cast<const uint32_t*>(ptr + sizeof(uint32_t));
  auto values  = ptr + sizeof(uint32_t) + sizeof(uint32_t) * num_values;

  for (uint32_t idx = 0, offset = 0; idx < num_values; ++idx) {
    const auto next_offset = offsets[idx];
    const auto size        = next_offset - offset;
    return_values_.push_back(ReturnValue::unpack(values + offset, size, mem_kind));
    offset = next_offset;
  }
}

/*static*/ ReturnValue ReturnValues::extract(const Legion::Future& future, uint32_t to_extract)
{
  auto kind          = find_memory_kind_for_executing_processor();
  const auto* buffer = future.get_buffer(kind);

  auto ptr        = static_cast<const int8_t*>(buffer);
  auto num_values = *reinterpret_cast<const uint32_t*>(ptr);

  auto offsets = reinterpret_cast<const uint32_t*>(ptr + sizeof(uint32_t));
  auto values  = ptr + sizeof(uint32_t) + sizeof(uint32_t) * num_values;

  const uint32_t next_offset = offsets[to_extract];
  const uint32_t offset      = to_extract == 0 ? 0 : offsets[to_extract - 1];
  const uint32_t size        = next_offset - offset;

  return ReturnValue::unpack(values + offset, size, kind);
}

void ReturnValues::finalize(Legion::Context legion_context) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  }
  if (return_values_.size() == 1) {
    return_values_.front().finalize(legion_context);
    return;
  }

#if LegateDefined(LEGATE_USE_CUDA)
  auto kind = Processor::get_executing_processor().kind();
  // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
  //        as doing so would require the packing to be chained up with all preceding kernels,
  //        potentially launched with different streams, within the task. Until we find
  //        the right approach, we simply synchronize the device before proceeding.
  if (kind == Processor::TOC_PROC) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  const size_t return_size = legion_buffer_size();
  auto return_buffer =
    Legion::UntypedDeferredValue(return_size, find_memory_kind_for_executing_processor());
  const AccessorWO<int8_t, 1> acc{return_buffer, return_size, false};
  legion_serialize(acc.ptr(0));
  return_buffer.finalize(legion_context);
}

struct JoinReturnedException {
  using LHS = ReturnedException;
  using RHS = LHS;

  static const ReturnedException identity;

  template <bool EXCLUSIVE>
  static void apply(LHS& lhs, RHS rhs)
  {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(EXCLUSIVE);
    }
    if (lhs.raised() || !rhs.raised()) {
      return;
    }
    lhs = std::move(rhs);
  }

  template <bool EXCLUSIVE>
  static void fold(RHS& rhs1, RHS rhs2)
  {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(EXCLUSIVE);
    }
    if (rhs1.raised() || !rhs2.raised()) {
      return;
    }
    rhs1 = std::move(rhs2);
  }
};

/*static*/ const ReturnedException JoinReturnedException::identity;

namespace {

void pack_returned_exception(const ReturnedException& value, void*& ptr, size_t& size)
{
  auto new_size = value.legion_buffer_size();
  if (new_size > size) {
    size         = new_size;
    auto new_ptr = realloc(ptr, new_size);
    assert(new_ptr);  // realloc returns nullptr on failure, so check before clobbering ptr
    ptr = new_ptr;
  }
  value.legion_serialize(ptr);
}

}  // namespace

static void returned_exception_init(  // NOLINT(misc-use-anonymous-namespace)
  const Legion::ReductionOp* /*reduction_op*/,
  void*& ptr,
  size_t& size)
{
  pack_returned_exception(JoinReturnedException::identity, ptr, size);
}

static void returned_exception_fold(  // NOLINT(misc-use-anonymous-namespace)
  const Legion::ReductionOp* /*reduction_op*/,
  void*& lhs_ptr,
  size_t& lhs_size,
  const void* rhs_ptr)

{
  ReturnedException lhs, rhs;
  lhs.legion_deserialize(lhs_ptr);
  rhs.legion_deserialize(rhs_ptr);
  JoinReturnedException::fold<true>(lhs, rhs);
  pack_returned_exception(lhs, lhs_ptr, lhs_size);
}

void register_exception_reduction_op(const Library* library)
{
  auto redop_id = library->get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  auto* redop   = Realm::ReductionOpUntyped::create_reduction_op<JoinReturnedException>();
  Legion::Runtime::register_reduction_op(
    redop_id, redop, returned_exception_init, returned_exception_fold);
}

}  // namespace legate::detail
