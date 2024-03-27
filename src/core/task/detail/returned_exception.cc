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

#include "core/task/detail/returned_exception.h"

#include "core/runtime/detail/library.h"
#include "core/utilities/assert.h"
#include "core/utilities/detail/type_traits.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <utility>

namespace legate::traits::detail {

template <typename T, typename... Ts>
struct is_same_as_one_of<T, std::variant<Ts...>> : is_same_as_one_of<T, Ts...> {};

}  // namespace legate::traits::detail

namespace legate::detail {

template <typename T>
/*static*/ ReturnedException ReturnedException::construct_specific_from_buffer_(const void* buf)
{
  static_assert(traits::detail::is_same_as_one_of_v<T, variant_type>);

  auto ret = T{};

  ret.legion_deserialize(buf);
  return ret;
}

// ==========================================================================================

bool ReturnedException::raised() const
{
  return visit([&](auto&& rexn) { return rexn.raised(); });
}

std::size_t ReturnedException::legion_buffer_size() const
{
  return visit([](auto&& rexn) { return rexn.legion_buffer_size(); });
}

void ReturnedException::legion_serialize(void* buffer) const
{
  visit([&](auto&& rexn) { rexn.legion_serialize(buffer); });
}

void ReturnedException::legion_deserialize(const void* buffer)
{
  *this = ReturnedException::construct_from_buffer(buffer);
}

ReturnValue ReturnedException::pack() const
{
  return visit([](auto&& rexn) { return rexn.pack(); });
}

std::string ReturnedException::to_string() const
{
  return visit([&](auto&& rexn) { return rexn.to_string(); });
}

void ReturnedException::throw_exception()
{
  visit([&](auto&& rexn) {
    LegateAssert(rexn.raised());
    rexn.throw_exception();
  });
  LegateUnreachable();
}

/*static*/ ReturnedException ReturnedException::construct_from_buffer(const void* buf)
{
  // If alignment isn't 1, then we cannot just static_cast below, since we might need to
  // align the pointer first...
  static_assert(alignof(ExceptionKind) == 1);
  const auto kind = *static_cast<const ExceptionKind*>(buf);

  switch (kind) {
    case ExceptionKind::CPP: return construct_specific_from_buffer_<ReturnedCppException>(buf);
    case ExceptionKind::PYTHON:
      return construct_specific_from_buffer_<ReturnedPythonException>(buf);
  }
  LEGATE_ABORT("Unhandled exception kind: " << static_cast<int>(kind));
  LegateUnreachable();
}

// ==========================================================================================

class JoinReturnedException {
 public:
  using LHS = ReturnedException;
  using RHS = LHS;

  static const ReturnedException identity;

  template <bool EXCLUSIVE>
  static void apply(LHS& lhs, RHS rhs)
  {
    do_op<EXCLUSIVE>(lhs, std::move(rhs));
  }

  template <bool EXCLUSIVE>
  static void fold(RHS& rhs1, RHS rhs2)
  {
    do_op<EXCLUSIVE>(rhs1, std::move(rhs2));
  }

 private:
  template <bool EXCLUSIVE>
  static void do_op(LHS& lhs, RHS rhs)
  {
    LegateCheck(EXCLUSIVE);
    if (lhs.raised() || !rhs.raised()) {
      return;
    }
    lhs = std::move(rhs);
  }
};

/*static*/ const ReturnedException JoinReturnedException::identity{};

namespace {

void pack_returned_exception(const ReturnedException& value, void** ptr, std::size_t* size)
{
  if (const auto new_size = value.legion_buffer_size(); new_size > *size) {
    const auto new_ptr = std::realloc(*ptr, new_size);
    // realloc returns nullptr on failure, so check before clobbering ptr
    LegateCheck(new_ptr);
    *size = new_size;
    *ptr  = new_ptr;
  }
  value.legion_serialize(*ptr);
}

void returned_exception_init(const Legion::ReductionOp* /*reduction_op*/,
                             void*& ptr,
                             std::size_t& size)
{
  pack_returned_exception(JoinReturnedException::identity, &ptr, &size);
}

void returned_exception_fold(const Legion::ReductionOp* /*reduction_op*/,
                             void*& lhs_ptr,
                             std::size_t& lhs_size,
                             const void* rhs_ptr)
{
  auto lhs = ReturnedException::construct_from_buffer(lhs_ptr);
  auto rhs = ReturnedException::construct_from_buffer(rhs_ptr);

  JoinReturnedException::fold<true>(lhs, rhs);
  pack_returned_exception(lhs, &lhs_ptr, &lhs_size);
}

}  // namespace

void register_exception_reduction_op(const Library* library)
{
  const auto redop_id = library->get_reduction_op_id(LEGATE_CORE_JOIN_EXCEPTION_OP);
  auto* redop         = Realm::ReductionOpUntyped::create_reduction_op<JoinReturnedException>();
  Legion::Runtime::register_reduction_op(
    redop_id, redop, returned_exception_init, returned_exception_fold);
}

}  // namespace legate::detail
