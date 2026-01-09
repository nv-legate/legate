/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/redop/base.h>
#include <legate/redop/detail/base.h>
#include <legate/redop/detail/complex.h>
#include <legate/type/complex.h>
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
 * @brief Sum reduction specialization for `Complex<Half>`.
 */
template <>
class LEGATE_EXPORT SumReduction<Complex<Half>>
  : public detail::
      BaseReduction<Complex<Half>, ReductionOpKind::ADD, detail::AtomicWrapperComplexHalf> {
 public:
  /**
   * @brief Identity value for the sum reduction (zero-initialized).
   */
  static inline const auto identity =  // NOLINT(readability-identifier-naming, cert-err58-cpp)
    value_type{};

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
 * @brief Multiply reduction specialization for `Complex<Half>`.
 */
template <>
class LEGATE_EXPORT ProdReduction<Complex<Half>>
  : public detail::
      BaseReduction<Complex<Half>, ReductionOpKind::MUL, detail::AtomicWrapperComplexHalf> {
 public:
  /**
   * @brief Identity value for the multiplication reduction.
   */
  static inline const auto identity =  // NOLINT(readability-identifier-naming, cert-err58-cpp)
    value_type{Half{1.F}};

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

// ==========================================================================================

/**
 * @brief Sum reduction specialization for `Complex<float>`.
 */
template <>
class LEGATE_EXPORT SumReduction<Complex<float>>
  : public detail::BaseReduction<Complex<float>, ReductionOpKind::ADD> {
 public:
  /**
   * @brief Identity value for the sum reduction (zero-initialized).
   */
  static constexpr auto identity =  // NOLINT(readability-identifier-naming)
    value_type{};

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
 * @brief Multiply reduction specialization for `Complex<float>`.
 */
template <>
class LEGATE_EXPORT ProdReduction<Complex<float>>
  : public detail::BaseReduction<Complex<float>, ReductionOpKind::MUL> {
 public:
  /**
   * @brief Identity value for the multiplication reduction.
   */
  static constexpr auto identity =  // NOLINT(readability-identifier-naming)
    value_type{1.};

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

// ==========================================================================================

/**
 * @brief Sum reduction specialization for `Complex<Half>`.
 */
template <>
class LEGATE_EXPORT SumReduction<Complex<double>>
  : public detail::
      BaseReduction<Complex<double>, ReductionOpKind::ADD, detail::AtomicWrapperComplexDouble> {
 public:
  /**
   * @brief Identity value for the sum reduction (zero-initialized).
   */
  static constexpr auto identity =  // NOLINT(readability-identifier-naming)
    value_type{};

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

#define LEGATE_FOREACH_COMPLEX64_REDOP(__op__, ...)                            \
  do {                                                                         \
    __op__(legate::SumReduction<legate::Complex<legate::Half>>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<legate::Complex<legate::Half>>, __VA_ARGS__); \
  } while (0)

#define LEGATE_FOREACH_COMPLEX128_REDOP(__op__, ...)                    \
  do {                                                                  \
    __op__(legate::SumReduction<legate::Complex<float>>, __VA_ARGS__);  \
    __op__(legate::ProdReduction<legate::Complex<float>>, __VA_ARGS__); \
    __op__(legate::SumReduction<legate::Complex<double>>, __VA_ARGS__); \
  } while (0)

#define LEGATE_FOREACH_COMPLEX_REDOP(...)         \
  do {                                            \
    LEGATE_FOREACH_COMPLEX64_REDOP(__VA_ARGS__);  \
    LEGATE_FOREACH_COMPLEX128_REDOP(__VA_ARGS__); \
  } while (0)

#define LEGATE_FOREACH_SPECIALIZED_COMPLEX_REDOP(...) LEGATE_FOREACH_COMPLEX_REDOP(__VA_ARGS__)

}  // namespace legate
