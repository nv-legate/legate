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

#pragma once

#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace legate::detail {

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief Given an N-Dimensional shape and a point inside that shape, compute the "linearized"
 * index of the point within the shape.
 *
 * @param lo The lowest point in the shape.
 * @param hi The highest point in the shape.
 * @param point The point whose position in the shape you wish to linearize.
 *
 * @return The linear index of the point.
 *
 * This routine is often used to determine the "local", 0-based position of a point within a
 * task, that will be in the range `[0, shape.volume() - 1)`. This may be used to e.g. copy a
 * sub-store into a temporary 1D buffer, in which case `linearize()` would map each point in
 * the shape to an index within the buffer:
 * @code{.cpp}
 * auto shape = store.shape<DIM>();
 * auto *buf  = new int[shape.volume()];
 *
 * for (auto it = legate::PointInRectIterator<DIM>{shape}; it.valid(); ++it) {
 *   auto local_idx = legate::linearize(shape.lo, shape.hi, *it);
 *   // local_idx contains the 0-based index of *it, regardless of how the task was
 *   // parallelized
 *   buf[local_idx] = accessor[*it];
 * }
 * @endcode
 *
 * For example, given a 2x2 shape with bounds `lo` of `(0, 0)` and `hi` of `(2, 2)`, then
 * for each `point` the linearized indices would be as follows:
 * @code{.unparsed}
 * Point  -> idx
 * (0, 0) -> 0
 * (0, 1) -> 1
 * (0, 2) -> 2
 * (1, 0) -> 3
 * (1, 1) -> 4
 * (1, 2) -> 5
 * (2, 0) -> 6
 * (2, 1) -> 7
 * (2, 2) -> 8
 * @endcode
 * Similary, with a `lo` of `(2, 2)` and `hi` of `(4, 4)`:
 * @code{.unparsed}
 * Point  -> idx
 * (2, 2) -> 0
 * (2, 3) -> 1
 * (2, 4) -> 2
 * (3, 2) -> 3
 * (3, 3) -> 4
 * (3, 4) -> 5
 * (4, 2) -> 6
 * (4, 3) -> 7
 * (4, 4) -> 8
 * @endcode
 *
 * @see delinearize
 */
[[nodiscard]] std::size_t linearize(const DomainPoint& lo,
                                    const DomainPoint& hi,
                                    const DomainPoint& point);

/**
 * @brief Given an N-Dimensional shape and an index corresponding to a point inside that shape,
 * compute the point corresponding to the index.
 *
 * @param lo The lowest point in the shape.
 * @param hi The highest point in the shape.
 * @param idx The linearized index of the point.
 *
 * @return The point inside the shape.
 *
 * This routine is often used to convert a "local" 1d index, in the range `[0, shape.volume() -
 * 1)`, to a point within the "local" shape. For example, this is often used to convert a
 * thread ID in a CUDA kernel or OpenMP loop to the corresponding point within the shape:
 * @code{.cpp}
 * // e.g. in an OpenMP loop
 * auto shape = store.shape<DIM>();
 *
 * #omp parallel for
 * for (std::size_t i = 0; i < shape.volume(); ++i) {
 *   auto local_pt = legate::delinearize(shape.lo, shape.hi, i);
 *   // local_pt now contains the local point corresponding to index i
 * }
 * @endcode
 *
 * For example, given a 2x2 shape with bounds `lo` of `(0, 0)` and `hi` of `(2, 2)`, then
 * for each `idx`, the delinearized points would be as follows:
 * @code{.unparsed}
 * idx -> Point
 * 0   -> (0, 0)
 * 1   -> (0, 1)
 * 2   -> (0, 2)
 * 3   -> (1, 0)
 * 4   -> (1, 1)
 * 5   -> (1, 2)
 * 6   -> (2, 0)
 * 7   -> (2, 1)
 * 8   -> (2, 2)
 * @endcode
 *
 * @see linearize
 */
[[nodiscard]] DomainPoint delinearize(const DomainPoint& lo,
                                      const DomainPoint& hi,
                                      std::size_t idx);

/** @} */

}  // namespace legate::detail
