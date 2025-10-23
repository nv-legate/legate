/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/half.h>

#include <legate_defines.h>

#include <legate/cuda/detail/module_manager.h>
#include <legate/utilities/macros.h>

#include <realm/cuda/cuda_module.h>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <legate/generated/fatbin/redop_half_fatbin.h>
#else
namespace legate::detail {

namespace {

constexpr const void* redop_half_fatbin = nullptr;  // NOLINT(readability-identifier-naming)

}  // namespace

}  // namespace legate::detail
#endif

namespace legate {

/*static*/ void SumReduction<Half>::fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                                                    Realm::Cuda::CudaRedOpDesc* desc)
{
  desc->redop_id      = static_cast<Realm::ReductionOpID>(REDOP_ID);
  desc->apply_excl    = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                        "legate_sum_redop_half_apply_excl");
  desc->apply_nonexcl = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                           "legate_sum_redop_half_apply_non_excl");
  desc->fold_excl     = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                       "legate_sum_redop_half_fold_excl");
  desc->fold_nonexcl  = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                          "legate_sum_redop_half_fold_non_excl");
}

/*static*/ void ProdReduction<Half>::fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                                                     Realm::Cuda::CudaRedOpDesc* desc)
{
  desc->redop_id      = static_cast<Realm::ReductionOpID>(REDOP_ID);
  desc->apply_excl    = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                        "legate_prod_redop_half_apply_excl");
  desc->apply_nonexcl = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                           "legate_prod_redop_half_apply_non_excl");
  desc->fold_excl     = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                       "legate_prod_redop_half_fold_excl");
  desc->fold_nonexcl  = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                          "legate_prod_redop_half_fold_non_excl");
}

/*static*/ void MaxReduction<Half>::fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                                                    Realm::Cuda::CudaRedOpDesc* desc)
{
  desc->redop_id      = static_cast<Realm::ReductionOpID>(REDOP_ID);
  desc->apply_excl    = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                        "legate_max_redop_half_apply_excl");
  desc->apply_nonexcl = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                           "legate_max_redop_half_apply_non_excl");
  desc->fold_excl     = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                       "legate_max_redop_half_fold_excl");
  desc->fold_nonexcl  = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                          "legate_max_redop_half_fold_non_excl");
}

/*static*/ void MinReduction<Half>::fill_redop_desc(cuda::detail::CUDAModuleManager* manager,
                                                    Realm::Cuda::CudaRedOpDesc* desc)
{
  desc->redop_id      = static_cast<Realm::ReductionOpID>(REDOP_ID);
  desc->apply_excl    = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                        "legate_min_redop_half_apply_excl");
  desc->apply_nonexcl = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                           "legate_min_redop_half_apply_non_excl");
  desc->fold_excl     = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                       "legate_min_redop_half_fold_excl");
  desc->fold_nonexcl  = manager->load_function_from_fatbin(detail::redop_half_fatbin,
                                                          "legate_min_redop_half_fold_non_excl");
}

}  // namespace legate
