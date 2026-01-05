/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/redop/base.h>
#include <legate/redop/detail/base.h>
#include <legate/type/half.h>
#include <legate/type/types.h>

namespace Realm::Cuda {  // NOLINT

struct CudaRedOpDesc;

}  // namespace Realm::Cuda

namespace legate::cuda::detail {

class CUDAModuleManager;

}  // namespace legate::cuda::detail

namespace legate {

/**
 * @brief Sum reduction specialization for `Half`.
 */
template <>
class LEGATE_EXPORT SumReduction<Half> : public detail::BaseReduction<Half, ReductionOpKind::ADD> {
 public:
  /**
   * @brief Identity value for the sum reduction (zero-initialized).
   */
  static constexpr auto identity = value_type{};  // NOLINT(readability-identifier-naming)

  /**
   * @brief Fills a CUDA reduction operator descriptor for this reduction.
   *
   * This is a private routine and is not generally useful for users.
   *
   * @param manager Pointer to the CUDA module manager.
   * @param desc Pointer to the `CudaRedOpDesc` to populate.
   */
  static void fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                              Realm::Cuda::CudaRedOpDesc* desc);
};

/**
 * @brief Multiply reduction specialization for `Half`.
 */
template <>
class LEGATE_EXPORT ProdReduction<Half> : public detail::BaseReduction<Half, ReductionOpKind::MUL> {
 public:
  /**
   * @brief Identity value for the reduction.
   */
  static inline const auto identity =  // NOLINT(readability-identifier-naming, cert-err58-cpp)
    value_type{1.F};

  /**
   * @brief Fills a CUDA reduction operator descriptor for this reduction.
   *
   * This is a private routine and is not generally useful for users.
   *
   * @param manager Pointer to the CUDA module manager.
   * @param desc Pointer to the `CudaRedOpDesc` to populate.
   */
  static void fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                              Realm::Cuda::CudaRedOpDesc* desc);
};

/**
 * @brief Max reduction specialization for `Half`.
 */
template <>
class LEGATE_EXPORT MaxReduction<Half> : public detail::BaseReduction<Half, ReductionOpKind::MAX> {
 public:
  /**
   * @brief Identity value for the reduction.
   */
  static inline const auto identity =  // NOLINT(readability-identifier-naming, cert-err58-cpp)
    value_type{LEGATE_MIN_HALF};

  /**
   * @brief Fills a CUDA reduction operator descriptor for this reduction.
   *
   * This is a private routine and is not generally useful for users.
   *
   * @param manager Pointer to the CUDA module manager.
   * @param desc Pointer to the `CudaRedOpDesc` to populate.
   */
  static void fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                              Realm::Cuda::CudaRedOpDesc* desc);
};

/**
 * @brief Min reduction specialization for `Half`.
 */
template <>
class LEGATE_EXPORT MinReduction<Half> : public detail::BaseReduction<Half, ReductionOpKind::MIN> {
 public:
  /**
   * @brief Identity value for the reduction.
   */
  static inline const auto identity =  // NOLINT(readability-identifier-naming, cert-err58-cpp)
    value_type{LEGATE_MAX_HALF};

  /**
   * @brief Fills a CUDA reduction operator descriptor for this reduction.
   *
   * This is a private routine and is not generally useful for users.
   *
   * @param manager Pointer to the CUDA module manager.
   * @param desc Pointer to the `CudaRedOpDesc` to populate.
   */
  static void fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                              Realm::Cuda::CudaRedOpDesc* desc);
};

#define LEGATE_FOREACH_FLOAT16_REDOP(__op__, ...)             \
  do {                                                        \
    __op__(legate::SumReduction<legate::Half>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<legate::Half>, __VA_ARGS__); \
    __op__(legate::MaxReduction<legate::Half>, __VA_ARGS__);  \
    __op__(legate::MinReduction<legate::Half>, __VA_ARGS__);  \
  } while (0)

#define LEGATE_FOREACH_SPECIALIZED_FLOAT16_REDOP(...) LEGATE_FOREACH_FLOAT16_REDOP(__VA_ARGS__)

}  // namespace legate
