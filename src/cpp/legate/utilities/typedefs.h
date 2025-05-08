/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

/**
 * @file
 * @brief Type aliases to Legion components
 */

namespace legate {

class TaskContext;

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief Function signature for task variants. Each task variant must be a function of this type.
 */
using VariantImpl = void (*)(TaskContext);

/**
 * @brief Function signature for direct-to-legion task variants. Users should usually prefer
 * VariantImpl instead.
 */
template <typename T = void>
using LegionVariantImpl = T (*)(const Legion::Task*,
                                const std::vector<Legion::PhysicalRegion>&,
                                Legion::Context,
                                Legion::Runtime*);

/**
 * @brief Signature for a callable to be executed right before the runtime shuts down.
 */
using ShutdownCallback = std::function<void(void)>;

/**
 * @brief An enum describing the kind of variant.
 *
 * @note The values don't start at 0. This is to match Legion, where `0` is the 'None' variant.
 */
enum class VariantCode : Legion::VariantID {  // NOLINT(performance-enum-size)
  CPU = 1,                                    ///< A CPU variant.
  GPU,                                        ///< A GPU variant.
  OMP                                         ///< An OpenMP variant.
};

using LegateVariantCode [[deprecated("since 24.11: use legate::VariantCode instead")]] =
  VariantCode;

// Re-export Legion types
using Legion::Logger;
using Legion::TunableID;

/**
 * @brief Integer type representing a `Library`-local task ID.
 *
 * All tasks are uniquely identifiable via a "task ID". These task ID's come in 2 flavors:
 * global and local. When a task is registered to a `Library`, the task must declare a unique
 * "local" task ID (`LocalTaskID`) within that `Library`. This task ID must not coincide with
 * any other task ID within that `Library`. After registration, the task is also assigned a
 * "global" ID (`GlobalTaskID`) which is guaranteed to be unique across the entire program.
 *
 * `GlobalTaskID`s may therefore be used to refer to tasks registered to other `Library`s or to
 * refer to the task when interfacing with Legion.
 *
 * For example, consider a task `Foo`:
 * @snippet unit/library.cc Foo declaration
 * And two `Library`s, `bar` and `baz`:
 * @snippet unit/library.cc TaskID registration
 *
 * @see GlobalTaskID Library Library::get_task_id()
 */
enum class LocalTaskID : std::int64_t {};

/**
 * @brief Integer type representing a global task ID.
 *
 * `GlobalTaskID`s may be used to refer to tasks registered to other `Library`s or to refer to
 * the task when interfacing with Legion. See `LocalTaskID` for further discussion on task ID's
 * and task registration.
 *
 * @see LocalTaskID Library Library::get_local_task_id()
 */
enum class GlobalTaskID : Legion::TaskID {};

/**
 * @brief Integer type representing a `Library`-local reduction operator ID.
 *
 * All reduction operators are uniquely identifiable via a "reduction ID", which serve as proxy
 * task ID's for the reduction meta-tasks. When a reduction operator is registered with a
 * `Library`, the reduction must declare a unique "local" ID (`LocalRedopID`) within that
 * `Library`. The `Library` then assigns a globally unique ID to the reduction operator, which
 * may be used to refer to the operator across the entire program.
 *
 * @see GlobalRedopID Library Library::get_reduction_op_id()
 */
enum class LocalRedopID : std::int64_t {};

/**
 * @brief Integer type representing a global reduction operator ID.
 *
 * `GlobalRedopID`s may be used to refer to reduction operators registered to other `Library`s,
 * or to refer to the reduction operator when interfacing with Legion. See `LocalRedopID` for
 * further discussion on reduction operator ID's.
 *
 * @see LocalRedopID
 */
enum class GlobalRedopID : Legion::ReductionOpID {};

/** @} */  // end of util

// Geometry types

/**
 * @addtogroup geometry
 * @{
 */

/**
 * @brief Coordinate type.
 */
using Legion::coord_t;

/**
 * @brief Type for multi-dimensional points.
 *
 * Point objects support index expressions; they can be accessed like a statically-sized array.
 * Point objects also support usual arithmetic operators and a dot operator.
 *
 * For a complete definition, see
 * [Realm::Point](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L46-L124).
 */
template <int DIM, typename T = coord_t>
using Point = Legion::Point<DIM, T>;

/**
 * @brief Type for multi-dimensional rectangles.
 *
 * Each rectangle consists of two legate::Point objects, one for the lower
 * bounds (`.lo`) and one for the upper bounds (`.hi`).
 *
 * For a complete definition, see
 * [Realm::Rect](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L126-L212).
 */
template <int DIM, typename T = coord_t>
using Rect = Legion::Rect<DIM, T>;

/**
 * @brief Dimension-erased type for multi-dimensional points.
 *
 * For a complete definition, see
 * [Legion::DomainPoint](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L127-L253).
 */
using Legion::DomainPoint;

/**
 * @brief Dimension-erased type for multi-dimensional rectangles.
 *
 * For a complete definition, see
 * [Legion::Domain](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L255-L543).
 */
using Legion::Domain;

/** @} */  // end of geometry

// Accessor types

/**
 * @addtogroup accessor
 * @{
 */

/**
 * @brief Read-only accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2555-L2562)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorRO =
  Legion::FieldAccessor<LEGION_READ_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Write-only accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2575-L2581)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorWO =
  Legion::FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Read-write accessor
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2564-L2573)
 * for a complete list of supported operators.
 */
template <typename FT, int N, typename T = coord_t>
using AccessorRW =
  Legion::FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>>;

/**
 * @brief Reduction accessor
 *
 * Unlike the other accessors, an index expression on a reduction accessor allows the client to
 * perform only two operations, `<<=` and `reduce`, both of which reduce a value to the chosen
 * element.
 *
 * See
 * [legion.h](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion.h#L2837-L2848)
 * for details about the reduction accessor.
 */
template <typename REDOP, bool EXCLUSIVE, int N, typename T = coord_t>
using AccessorRD = Legion::
  ReductionAccessor<REDOP, EXCLUSIVE, N, T, Realm::AffineAccessor<typename REDOP::RHS, N, T>>;

/** @} */  // end of accessor

// Iterators

/**
 * @addtogroup iterator
 * @{
 */

/**
 * @brief Iterator that iterates all points in a given `legate::Rect`.
 *
 * See
 * [Realm::PointInRectIterator](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/point.h#L239-L255)
 * for a complete definition.
 */
template <int DIM, typename T = coord_t>
using PointInRectIterator = Legion::PointInRectIterator<DIM, T>;

/**
 * @brief Iterator that iterates all points in a given `legate::Domain`.
 *
 * See
 * [Legion::PointInDomainIterator](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_domain.h#L599-L622)
 * for a complete definition.
 */
template <int DIM, typename T = coord_t>
using PointInDomainIterator = Legion::PointInDomainIterator<DIM, T>;

/** @} */  // end of iterator

// Machine

/**
 * @addtogroup machine
 * @{
 */

/**
 * @brief Logical processor handle
 *
 * Legate libraries rarely use processor handles directly and there are no Legate APIs that
 * take a processor handle. However, the libraries may want to query the processor that runs
 * the current task to perform some processor- or processor kind-specific operations. In that
 * case, `legate::Runtime::get_runtime().get_executing_processor()` can be used. Other useful
 * memobers of `legate::Processor` are the `kind` method, which returns the processor kind, and
 * `legate::Processor::Kind`, an enum for all processor types.
 *
 * See
 * [Realm::Processor](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/processor.h#L35-L141)
 * for a complete definition. The list of processor types can be found
 * [here](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/realm_c.h#L45-L54).
 *
 */
using Legion::Processor;

/**
 * @brief Logical memory handle
 *
 * In Legate, libraries will never have to use memory handles directly. However, some Legate
 * APIs (e.g., \ref create_buffer()) take a memory kind as an argument; `legate::Memory::Kind`
 * is an enum for all memory types.
 *
 * See
 * [Realm::Memory](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/memory.h#L30-L65)
 * for a complete definition. The list of memory types can be found
 * [here](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/realm/realm_c.h#L63-L78).
 */
using Legion::Memory;

/** @} */  // end of machine

// Reduction operators

/**
 * @addtogroup reduction
 * @{
 */

/**
 * @brief Reduction with addition
 *
 * See
 * [Legion::SumReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L46-L285).
 */
template <typename T>
using SumReduction = Legion::SumReduction<T>;

/**
 * @brief Reduction with subtraction
 *
 * See
 * [Legion::DiffReduction](https://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L287-L492).
 */
template <typename T>
using DiffReduction = Legion::DiffReduction<T>;

/**
 * @brief Reduction with multiplication
 *
 * See
 * [Legion::ProdReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L494-L714).
 */
template <typename T>
using ProdReduction = Legion::ProdReduction<T>;

/**
 * @brief Reduction with division
 *
 * See
 * [Legion::DivReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L716-L921).
 */
template <typename T>
using DivReduction = Legion::DivReduction<T>;

/**
 * @brief Reduction with the binary max operator
 *
 * See
 * [Legion::MaxReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L923-L1109).
 */
template <typename T>
using MaxReduction = Legion::MaxReduction<T>;

/**
 * @brief Reduction with the binary min operator
 *
 * See
 * [Legion::MinReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1111-L1297).
 */
template <typename T>
using MinReduction = Legion::MinReduction<T>;

/**
 * @brief Reduction with bitwise or
 *
 * See
 * [Legion::OrReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1299-L1423).
 */
template <typename T>
using OrReduction = Legion::OrReduction<T>;

/**
 * @brief Reduction with bitwise and
 *
 * See
 * [Legion::AndReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1425-L1549).
 */
template <typename T>
using AndReduction = Legion::AndReduction<T>;

/**
 * @brief Reduction with bitwise xor
 *
 * See
 * [Legion::XorReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1551-L1690).
 */
template <typename T>
using XorReduction = Legion::XorReduction<T>;

/** @} */  // end of reduction

namespace detail {

[[nodiscard]] Logger& log_legate();

[[nodiscard]] Logger& log_legate_partitioner();

}  // namespace detail

}  // namespace legate

// Until Realm does this for us, we declare this here
namespace std {

template <>
struct hash<legate::Memory> {
  [[nodiscard]] std::size_t operator()(const legate::Memory& mem) const noexcept
  {
    using id_type = decltype(mem.id);
    return std::hash<id_type>{}(mem.id);
  }
};

}  // namespace std

// backwards-compat workaround, should not use
[[deprecated("since 24.11: using legate::VariantCode::CPU instead")]] inline constexpr auto
  LEGATE_CPU_VARIANT = legate::VariantCode::CPU;
[[deprecated("since 24.11: using legate::VariantCode::GPU instead")]] inline constexpr auto
  LEGATE_GPU_VARIANT = legate::VariantCode::GPU;
[[deprecated("since 24.11: using legate::VariantCode::OMP instead")]] inline constexpr auto
  LEGATE_OMP_VARIANT = legate::VariantCode::OMP;
