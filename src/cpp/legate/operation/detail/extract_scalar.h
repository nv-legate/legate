/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/task/detail/legion_task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/macros.h>

#include <legion/legion_types.h>

#include <vector>

// Doxygen chokes on this with
//
// /src/cpp/legate/operation/detail/extract_scalar.h:34: error: documented symbol 'void
// Legion::LegionSerialization::end_task< UntypedDeferredValue>' was not declared or defined.
//
// Because it thinks we are documenting a symbol we defined somewhere else (we aren't).
#ifndef DOXYGEN
namespace Legion {

// Legion defines this specialization for DeferredValue<T> but not UntypedDeferredValue. This
// is likely a mistake, since calling `result.finalize()` on its own seems to work. Trying to
// use the normal Legion::LegionTaskWrapper::task_wrapper mechanisms lead to
//
// [0 - 17064b000] 0.388027 {5}{runtime}: LEGION ERROR: Task legate::detail::(anonymous
// namespace)::ExtractScalar (UID 19) used a task variant with a maximum return size of 1 but
// returned a result of 8 bytes. (from file /legion-src/runtime/legion/legion_tasks.cc:4626)
//
// Because it incorrectly computes the return size based on sizeof(UntypedDeferredValue).
//
// This is fixed by
// https://gitlab.com/StanfordLegion/legion/-/commit/1be420ecad2167fda7d1879ea8f17eeed1e43fed,
// but until we bump Legion for that hash, we define it ourselves here. If and when we do bump,
// we should get compile errors from duplicate definitions, so we'll know to remove this.
template <bool FINAL>
struct LegionSerialization::NonPODSerializer<UntypedDeferredValue, false, FINAL> {
  static void end_task(Context ctx, UntypedDeferredValue* result) { result->finalize(ctx); }

  static Future from_value(const UntypedDeferredValue*) { std::abort(); }

  static UntypedDeferredValue unpack(const Future&, bool, const char*) { std::abort(); }
};

}  // namespace Legion
#endif

namespace legate::detail {

class ExtractScalar : public LegionTask<ExtractScalar> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{LocalTaskID{CoreTask::EXTRACT_SCALAR}}
      .with_variant_options(
        legate::VariantOptions{}.with_has_allocations(true).with_elide_device_ctx_sync(true))
      .with_signature(legate::TaskSignature{}.inputs(0).outputs(0).scalars(2));

  [[nodiscard]] static Legion::UntypedDeferredValue cpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);

#if LEGATE_DEFINED(LEGATE_USE_OPENMP)
  [[nodiscard]] static Legion::UntypedDeferredValue omp_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);
#endif

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  [[nodiscard]] static Legion::UntypedDeferredValue gpu_variant(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions,
    Legion::Context context,
    Legion::Runtime* runtime);
#endif
};

}  // namespace legate::detail
