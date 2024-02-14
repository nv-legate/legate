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

#include "core/data/scalar.h"
#include "core/task/task_info.h"
#include "core/utilities/typedefs.h"

#include <memory>
#include <string>

/**
 * @file
 * @brief Class definition for legate::Library
 */

namespace legate::detail {
class Library;
}  // namespace legate::detail

namespace legate::mapping {
class Mapper;
}  // namespace legate::mapping

namespace legate {

class Runtime;

/**
 * @ingroup runtime
 * @brief A library class that provides APIs for registering components
 */
class Library {
 public:
  /**
   * @brief Returns the name of the library
   *
   * @return Library name
   */
  [[nodiscard]] const std::string& get_library_name() const;

  [[nodiscard]] Legion::TaskID get_task_id(std::int64_t local_task_id) const;
  [[nodiscard]] Legion::MapperID get_mapper_id() const;
  [[nodiscard]] Legion::ReductionOpID get_reduction_op_id(std::int64_t local_redop_id) const;
  [[nodiscard]] Legion::ProjectionID get_projection_id(std::int64_t local_proj_id) const;
  [[nodiscard]] Legion::ShardingID get_sharding_id(std::int64_t local_shard_id) const;

  [[nodiscard]] std::int64_t get_local_task_id(Legion::TaskID task_id) const;
  [[nodiscard]] std::int64_t get_local_reduction_op_id(Legion::ReductionOpID redop_id) const;
  [[nodiscard]] std::int64_t get_local_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] std::int64_t get_local_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] bool valid_task_id(Legion::TaskID task_id) const;
  [[nodiscard]] bool valid_reduction_op_id(Legion::ReductionOpID redop_id) const;
  [[nodiscard]] bool valid_projection_id(Legion::ProjectionID proj_id) const;
  [[nodiscard]] bool valid_sharding_id(Legion::ShardingID shard_id) const;

  [[nodiscard]] std::int64_t get_new_task_id();

  /**
   * @brief Returns the name of a task
   *
   * @param local_task_id Task id
   * @return Name of the task
   */
  [[nodiscard]] const std::string& get_task_name(std::int64_t local_task_id) const;
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
   *   __CUDA_HD__ inline static void apply(LHS& lhs, RHS rhs)
   *   {
   *     ...
   *   }
   *   template <bool EXCLUSIVE>
   *   __CUDA_HD__ inline static void fold(RHS& rhs1, RHS rhs2)
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
   * for each opeartor within the library.
   *
   * Finally, the contract for `apply` and `fold` is that they must update the
   * reference atomically when the `EXCLUSIVE` is `false`.
   *
   * @tparam REDOP Reduction operator to register
   * @param redop_id Library-local reduction operator ID
   *
   * @return Global reduction operator ID
   */
  template <typename REDOP>
  [[nodiscard]] std::int64_t register_reduction_operator(std::int32_t redop_id);
  /**
   * @brief Registers a library mapper. Replaces the existing mapper if there already is one.
   *
   * @param mapper A Legate mapper to register for the library
   */
  void register_mapper(std::unique_ptr<mapping::Mapper> mapper);

  void register_task(std::int64_t local_task_id, std::unique_ptr<TaskInfo> task_info);
  [[nodiscard]] const TaskInfo* find_task(std::int64_t local_task_id) const;

  Library() = default;
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
  static void perform_callback(Legion::RegistrationWithArgsCallbackFnptr callback,
                               const Legion::UntypedBuffer& buffer);

  detail::Library* impl_{};
};

}  // namespace legate

#include "core/runtime/library.inl"
