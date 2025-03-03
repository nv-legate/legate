/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/experimental/stl/detail/meta.hpp>
#include <legate/experimental/stl/detail/type_traits.hpp>
#include <legate/utilities/assert.h>

#include <iterator>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

/**
 * @cond
 */
namespace legate::experimental::stl {

// TODO(ericniebler)
// This should be Ignore instead, but I'm not changing it because it's part of the public API.
class ignore {  // NOLINT(readability-identifier-naming)
 public:
  ignore() = default;

  template <typename... Args>
  LEGATE_HOST_DEVICE constexpr ignore(Args&&...) noexcept  // NOLINT(google-explicit-constructor)
  {
  }
};

namespace detail {

template <typename LegateTask>
[[nodiscard]] LocalTaskID task_id_for(Library& library)
{
  static const LocalTaskID s_task_id = [&] {
    const LocalTaskID task_id = library.get_new_task_id();

    LegateTask::register_variants(library, task_id);
    return task_id;
  }();
  return s_task_id;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// A stateless lambda has a conversion to a function pointer.
template <typename Lambda>
using lambda_fun_ptr_t = decltype(+std::declval<Lambda>());

template <typename Lambda>
inline constexpr bool is_stateless_lambda_v = meta::evaluable_q<lambda_fun_ptr_t, Lambda>;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Function>
void check_function_type()
{
  using function_type = std::decay_t<Function>;

  static_assert(std::is_trivially_copyable_v<function_type>,
                "The function object is not trivially copyable.");

#if defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) && !defined(__CUDA_ARCH__)
  static_assert(!__nv_is_extended_device_lambda_closure_type(function_type),
                "Attempt to use an extended __device__ lambda in a context "
                "that requires querying its return type in host code. Use a "
                "named function object, a __host__ __device__ lambda, or "
                "cuda::proclaim_return_type instead.");
  static_assert(!__nv_is_extended_host_device_lambda_closure_type(function_type) ||
                  stateless_lambda<function_type>,
                "__host__ __device__ lambdas that have captures are not yet "
                "yet supported. Use a named function object instead.");
#endif
}

}  // namespace detail

template <typename Reference>
[[nodiscard]] Reference scalar_cast(const Scalar& scalar)
{
  using value_type = remove_cvref_t<Reference>;
  static_assert(std::is_trivially_copyable_v<value_type>);
  // Check to make sure the Scalar contains what we think it does.
  LEGATE_ASSERT(scalar.type().code() == Type::Code::BINARY);
  LEGATE_ASSERT(scalar.size() == sizeof(value_type));
  return *static_cast<const value_type*>(scalar.ptr());
}

template <typename Head, typename... Tail>
LEGATE_HOST_DEVICE [[nodiscard]] Head&& front_of(Head&& head, Tail&&... /*tail*/) noexcept
{
  return std::forward<Head>(head);
}

template <typename Task, typename... Parts>
void align_all(Task&, Parts&&...)
{
  static_assert(sizeof...(Parts) < 2);
}

template <typename Task, typename Part1, typename Part2, typename... OtherParts>
void align_all(Task& task, Part1&& part1, Part2&& part2, OtherParts&&... other_parts)
{
  task.add_constraint(legate::align(std::forward<Part1>(part1), part2));
  align_all(task, std::forward<Part2>(part2), std::forward<OtherParts>(other_parts)...);
}

template <typename AtLeastIteratorCategory, typename It>
void static_assert_iterator_category(const It&)
{
  static_assert(std::is_base_of_v<AtLeastIteratorCategory,
                                  typename std::iterator_traits<It>::iterator_category>);
}

}  // namespace legate::experimental::stl

/**
 * @endcond
 */

#include <legate/experimental/stl/detail/suffix.hpp>
