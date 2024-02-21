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

#pragma once

#include "core/utilities/assert.h"

#include "config.hpp"
#include "legate.h"
#include "meta.hpp"
#include "type_traits.hpp"

// Include this last:
#include "prefix.hpp"

namespace legate::stl {
struct ignore {
  ignore() = default;

  template <class... Args>
  LEGATE_STL_ATTRIBUTE((host, device))
  constexpr ignore(Args&&...) noexcept
  {
  }
};

namespace detail {

template <class LegateTask>
struct task_id_generator {
  std::int64_t operator()(Library& library) const
  {
    static const std::int64_t s_task_id = [&] {
      std::int64_t task_id = library.get_new_task_id();
      LegateTask::register_variants(library, task_id);
      return task_id;
    }();
    return s_task_id;
  }
};

template <class LegateTask>
inline constexpr task_id_generator<LegateTask> task_id_for{};

////////////////////////////////////////////////////////////////////////////////////////////////////
// A stateless lambda has a conversion to a function pointer.
template <class Lambda>
using lambda_fun_ptr_t = decltype(+std::declval<Lambda>());

template <class Lambda>
inline constexpr bool stateless_lambda = meta::evaluable_q<lambda_fun_ptr_t, Lambda>;

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class Function>
void check_function_type()
{
  static_assert(std::is_trivially_copyable_v<Function>,
                "The function object is not trivially copyable.");

#if defined(__NVCC__) && defined(__CUDACC_EXTENDED_LAMBDA__) && !defined(__CUDA_ARCH__)
  static_assert(!__nv_is_extended_device_lambda_closure_type(Function),
                "Attempt to use an extended __device__ lambda in a context "
                "that requires querying its return type in host code. Use a "
                "named function object, a __host__ __device__ lambda, or "
                "cuda::proclaim_return_type instead.");
  static_assert(
    !__nv_is_extended_host_device_lambda_closure_type(Function) || stateless_lambda<Function>,
    "__host__ __device__ lambdas that have captures are not yet "
    "yet supported. Use a named function object instead.");
#endif
}

}  // namespace detail

template <class Reference>
Reference scalar_cast(const Scalar& scalar)
{
  using value_type = remove_cvref_t<Reference>;
  static_assert(std::is_trivially_copyable_v<value_type>);
  // Check to make sure the Scalar contains what we think it does.
  LegateAssert(scalar.type().code() == Type::Code::BINARY);
  LegateAssert(scalar.size() == sizeof(value_type));
  return *static_cast<const value_type*>(scalar.ptr());
}

template <class Head, class... Tail>
LEGATE_STL_ATTRIBUTE((host, device))
Head&& front_of(Head&& head, Tail&&... /*tail*/) noexcept
{
  return std::forward<Head>(head);
}

template <class Task, class... Parts>
void align_all(Task& task, Parts&&...)
{
  static_assert(sizeof...(Parts) < 2);
}

template <class Task, class Part1, class Part2, class... OtherParts>
void align_all(Task& task, Part1&& part1, Part2&& part2, OtherParts&&... other_parts)
{
  task.add_constraint(legate::align(part1, part2));
  align_all(task, std::forward<Part2>(part2), std::forward<OtherParts>(other_parts)...);
}

}  // namespace legate::stl

#include "suffix.hpp"
