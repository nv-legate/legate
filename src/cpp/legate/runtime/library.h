/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/scalar.h>
#include <legate/task/task_info.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <map>
#include <string_view>

/**
 * @file
 * @brief Class definition for legate::Library
 */

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup runtime
 * @{
 */

class Runtime;

/**
 * @brief A library class that provides APIs for registering components
 */
class Library {
 public:
  /**
   * @brief Returns the name of the library
   *
   * @return Library name
   */
  [[nodiscard]] std::string_view get_library_name() const;

  [[nodiscard]] GlobalTaskID get_task_id(LocalTaskID local_task_id) const;
  [[nodiscard]] GlobalRedopID get_reduction_op_id(LocalRedopID local_redop_id) const;
  [[nodiscard]] Legion::ProjectionID get_projection_id(std::int64_t local_proj_id) const;
  [[nodiscard]] Legion::ShardingID get_sharding_id(std::int64_t local_shard_id) const;

  [[nodiscard]] LocalTaskID get_local_task_id(GlobalTaskID task_id) const;
  [[nodiscard]] LocalRedopID get_local_reduction_op_id(GlobalRedopID redop_id) const;
  [[nodiscard]] std::int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] std::int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] bool valid_task_id(GlobalTaskID task_id) const;
  [[nodiscard]] bool valid_reduction_op_id(GlobalRedopID redop_id) const;
  [[nodiscard]] bool valid_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] bool valid_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] LocalTaskID get_new_task_id();

  /**
   * @brief Returns the name of a task
   *
   * @param local_task_id Task id
   * @return Name of the task
   */
  [[nodiscard]] std::string_view get_task_name(LocalTaskID local_task_id) const;
  /**
   * @brief Retrieves a tunable parameter
   *
   * @param tunable_id ID of the tunable parameter
   * @param type Type of the tunable value
   *
   * @return The value of tunable parameter in a `Scalar`
   */
  [[nodiscard]] Scalar get_tunable(std::int64_t tunable_id, const Type& type);
  /**
   * @brief Registers a library specific reduction operator.
   *
   * The type parameter `REDOP` points to a class that implements a reduction operator.
   * Each reduction operator class has the following structure:
   *
   * @code{.cpp}
   * struct RedOp {
   *   using LHS = ...; // Type of the LHS values
   *   using RHS = ...; // Type of the RHS values
   *
   *   static const RHS identity = ...; // Identity of the reduction operator
   *
   *   template <bool EXCLUSIVE>
   *   LEGATE_HOST_DEVICE inline static void apply(LHS& lhs, RHS rhs)
   *   {
   *     ...
   *   }
   *   template <bool EXCLUSIVE>
   *   LEGATE_HOST_DEVICE inline static void fold(RHS& rhs1, RHS rhs2)
   *   {
   *     ...
   *   }
   * };
   * @endcode
   *
   * Semantically, Legate performs reductions of values `V0`, ..., `Vn` to element `E` in the
   * following way:
   *
   * @code{.cpp}
   * RHS T = RedOp::identity;
   * RedOp::fold(T, V0)
   * ...
   * RedOp::fold(T, Vn)
   * RedOp::apply(E, T)
   * @endcode
   * I.e., Legate gathers all reduction contributions using `fold` and applies the accumulator
   * to the element using `apply`.
   *
   * Oftentimes, the LHS and RHS of a reduction operator are the same type and `fold` and  `apply`
   * perform the same computation, but that's not mandatory. For example, one may implement
   * a reduction operator for subtraction, where the `fold` would sum up all RHS values whereas
   * the `apply` would subtract the aggregate value from the LHS.
   *
   * The reduction operator id (`REDOP_ID`) can be local to the library but should be unique
   * for each operator within the library.
   *
   * Finally, the contract for `apply` and `fold` is that they must update the
   * reference atomically when the `EXCLUSIVE` is `false`.
   *
   * @warning Because the runtime can capture the reduction operator and wrap it with CUDA
   * boilerplates only at compile time, the registration call should be made in a .cu file that
   * would be compiled by NVCC. Otherwise, the runtime would register the reduction operator in
   * CPU-only mode, which can degrade the performance when the program performs reductions on
   * non-scalar stores.
   *
   * @tparam REDOP Reduction operator to register
   * @param redop_id Library-local reduction operator ID
   *
   * @return Global reduction operator ID
   */
  template <typename REDOP>
  [[nodiscard]] GlobalRedopID register_reduction_operator(LocalRedopID redop_id);

  /**
   * @brief Register a task with the library.
   *
   * @param local_task_id The library-local task ID to assign for this task.
   * @param task_info The `TaskInfo` object describing the task.
   *
   * @throws std::out_of_range If the chosen local task ID exceeds the maximum local task ID
   * for the library.
   * @throws std::invalid_argument If the task (or another task with the same `local_task_id`)
   * has already been registered with the library.
   *
   * @see `find_task()`
   */
  void register_task(LocalTaskID local_task_id, const TaskInfo& task_info);

  /**
   * @brief Look up a task registered with the library.
   *
   * @param local_task_id The task ID to find.
   *
   * @return The `TaskInfo` object describing the task.
   *
   * @throws std::out_of_range If the task could not be found.
   *
   * @see `register_task()`
   */
  [[nodiscard]] TaskInfo find_task(LocalTaskID local_task_id) const;

  [[nodiscard]] const std::map<VariantCode, VariantOptions>& get_default_variant_options() const;

  Library() = LEGATE_DEFAULT_WHEN_CYTHON;

  explicit Library(detail::Library* impl);
  Library(const Library&)            = default;
  Library& operator=(const Library&) = default;
  Library(Library&&)                 = default;
  Library& operator=(Library&&)      = default;

  bool operator==(const Library& other) const;
  bool operator!=(const Library& other) const;

  [[nodiscard]] const detail::Library* impl() const;
  [[nodiscard]] detail::Library* impl();

 private:
  static void perform_callback_(Legion::RegistrationWithArgsCallbackFnptr callback,
                                const Legion::UntypedBuffer& buffer);

  detail::Library* impl_{};
};

/** @} */

}  // namespace legate

#include <legate/runtime/library.inl>
