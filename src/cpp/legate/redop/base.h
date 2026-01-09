/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// legion_redop.inl uses memcpy() but doesn't actually include the header for it.
#include <cstring>
//
#include <legate/utilities/detail/doxygen.h>

#include <legion/api/redop.h>

namespace legate {

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
class SumReduction : public Legion::SumReduction<T> {};

/**
 * @brief Reduction with multiplication
 *
 * See
 * [Legion::ProdReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L494-L714).
 */
template <typename T>
class ProdReduction : public Legion::ProdReduction<T> {};

/**
 * @brief Reduction with the binary max operator
 *
 * See
 * [Legion::MaxReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L923-L1109).
 */
template <typename T>
class MaxReduction : public Legion::MaxReduction<T> {};

/**
 * @brief Reduction with the binary min operator
 *
 * See
 * [Legion::MinReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1111-L1297).
 */
template <typename T>
class MinReduction : public Legion::MinReduction<T> {};

/**
 * @brief Reduction with bitwise or
 *
 * See
 * [Legion::OrReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1299-L1423).
 */
template <typename T>
class OrReduction : public Legion::OrReduction<T> {};

/**
 * @brief Reduction with bitwise and
 *
 * See
 * [Legion::AndReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1425-L1549).
 */
template <typename T>
class AndReduction : public Legion::AndReduction<T> {};

/**
 * @brief Reduction with bitwise xor
 *
 * See
 * [Legion::XorReduction](http://github.com/StanfordLegion/legion/blob/9ed6f4d6b579c4f17e0298462e89548a4f0ed6e5/runtime/legion/legion_redop.h#L1551-L1690).
 */
template <typename T>
class XORReduction : public Legion::XorReduction<T> {};

/** @} */  // end of reduction

#define LEGATE_FOREACH_BOOL_REDOP(__op__, ...)        \
  do {                                                \
    __op__(legate::SumReduction<bool>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<bool>, __VA_ARGS__); \
    __op__(legate::MaxReduction<bool>, __VA_ARGS__);  \
    __op__(legate::MinReduction<bool>, __VA_ARGS__);  \
    __op__(legate::OrReduction<bool>, __VA_ARGS__);   \
    __op__(legate::AndReduction<bool>, __VA_ARGS__);  \
    __op__(legate::XORReduction<bool>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_INT8_REDOP(__op__, ...)               \
  do {                                                       \
    __op__(legate::SumReduction<std::int8_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::int8_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::int8_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::int8_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::int8_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::int8_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::int8_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_INT16_REDOP(__op__, ...)               \
  do {                                                        \
    __op__(legate::SumReduction<std::int16_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::int16_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::int16_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::int16_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::int16_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::int16_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::int16_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_INT32_REDOP(__op__, ...)               \
  do {                                                        \
    __op__(legate::SumReduction<std::int32_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::int32_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::int32_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::int32_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::int32_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::int32_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::int32_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_INT64_REDOP(__op__, ...)               \
  do {                                                        \
    __op__(legate::SumReduction<std::int64_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::int64_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::int64_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::int64_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::int64_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::int64_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::int64_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_UINT8_REDOP(__op__, ...)               \
  do {                                                        \
    __op__(legate::SumReduction<std::uint8_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::uint8_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::uint8_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::uint8_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::uint8_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::uint8_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::uint8_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_UINT16_REDOP(__op__, ...)               \
  do {                                                         \
    __op__(legate::SumReduction<std::uint16_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::uint16_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::uint16_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::uint16_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::uint16_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::uint16_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::uint16_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_UINT32_REDOP(__op__, ...)               \
  do {                                                         \
    __op__(legate::SumReduction<std::uint32_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::uint32_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::uint32_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::uint32_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::uint32_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::uint32_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::uint32_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_UINT64_REDOP(__op__, ...)               \
  do {                                                         \
    __op__(legate::SumReduction<std::uint64_t>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<std::uint64_t>, __VA_ARGS__); \
    __op__(legate::MaxReduction<std::uint64_t>, __VA_ARGS__);  \
    __op__(legate::MinReduction<std::uint64_t>, __VA_ARGS__);  \
    __op__(legate::OrReduction<std::uint64_t>, __VA_ARGS__);   \
    __op__(legate::AndReduction<std::uint64_t>, __VA_ARGS__);  \
    __op__(legate::XORReduction<std::uint64_t>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_FLOAT32_REDOP(__op__, ...)      \
  do {                                                 \
    __op__(legate::SumReduction<float>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<float>, __VA_ARGS__); \
    __op__(legate::MaxReduction<float>, __VA_ARGS__);  \
    __op__(legate::MinReduction<float>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_FLOAT64_REDOP(__op__, ...)       \
  do {                                                  \
    __op__(legate::SumReduction<double>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<double>, __VA_ARGS__); \
    __op__(legate::MaxReduction<double>, __VA_ARGS__);  \
    __op__(legate::MinReduction<double>, __VA_ARGS__);  \
  } while (0)

}  // namespace legate
