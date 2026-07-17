/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

// This example validates the InlineStorage-backed `legate::Buffer` / `TaskLocalBuffer` classes, the
// `PhysicalStore::create_output_buffer` / `bind_data` implementations, and - crucially - multi-rank
// unbound stores. Each producer task is given an *unbound* output store (created with
// `create_store(type, dim)`, i.e. without a shape) and materializes its piece into a buffer that it
// binds back to the store. In a multi-rank run each rank produces its own piece; the pieces are
// stitched together (concatenated along the first dimension) to form the global store. To make the
// result independent of the number of ranks, each producer writes globally-consistent values: the
// element at global position `p` is `base + p` (the producer offsets by `task_index * num`). A
// checker then reads each now-bound store and verifies its contents, throwing on any mismatch so
// the failure is visible to the test runner via a non-zero exit code.

// Produces data by creating an output buffer that is bound to the store in one step
// (`create_output_buffer<T, DIM>(extents, /*bind_buffer=*/true)`). Also exercises a temporary,
// unbound scratch `create_buffer`.
class ProduceBoundBufferTask : public legate::LegateTask<ProduceBoundBufferTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true))
      .with_signature(legate::TaskSignature{}.outputs(2).scalars(2));

  static void cpu_variant(legate::TaskContext context);
};

// Produces data into a buffer created with `bind_buffer=false`, then binds it with a separate
// `bind_data(buffer, extents)` call.
class ProduceViaBindDataTask : public legate::LegateTask<ProduceViaBindDataTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true))
      .with_signature(legate::TaskSignature{}.outputs(2).scalars(2));

  static void cpu_variant(legate::TaskContext context);
};

// Produces data through the type-erased `TaskLocalBuffer` path
// (`create_output_buffer(DomainPoint, /*bind_buffer=*/true)`), converting it back to a typed
// `Buffer` to write into it.
class ProduceViaTaskLocalBufferTask : public legate::LegateTask<ProduceViaTaskLocalBufferTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{3}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true))
      .with_signature(legate::TaskSignature{}.outputs(2).scalars(2));

  static void cpu_variant(legate::TaskContext context);
};

// Reads a (now-bound) store with its natural partition and checks that the element at global
// position p equals base + p. In a multi-rank run each rank validates its own piece of the store.
// Throws on mismatch.
class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{4}}
      .with_variant_options(legate::VariantOptions{}.with_may_throw_exception(true))
      .with_signature(legate::TaskSignature{}.inputs(1).scalars(1));

  static void cpu_variant(legate::TaskContext context);
};

// Produces a 2D unbound store, exercising multi-dimensional buffer strides through the binding
// path. Pieces are concatenated along the rows; the element at global position (I, j) is set to
// base + I * cols + j.
class ProduceBound2DTask : public legate::LegateTask<ProduceBound2DTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{5}}
      .with_variant_options(legate::VariantOptions{}.with_has_allocations(true))
      .with_signature(legate::TaskSignature{}.outputs(2).scalars(3));

  static void cpu_variant(legate::TaskContext context);
};

// Reads a (now-bound) 2D store and checks that the element at global position (I, j) equals
// base + I * cols + j. Throws on mismatch.
class Check2DTask : public legate::LegateTask<Check2DTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{6}}
      .with_variant_options(legate::VariantOptions{}.with_may_throw_exception(true))
      .with_signature(legate::TaskSignature{}.inputs(1).scalars(1));

  static void cpu_variant(legate::TaskContext context);
};

// Reads the *whole* store replicated on every rank (via a broadcast constraint). This exercises the
// all-gather copy path out of the unbound store's Opaque partition into a replicated (NoPartition)
// target. Checks that the global store has the expected volume and that element p equals base + p.
// Throws on mismatch.
class CheckWholeTask : public legate::LegateTask<CheckWholeTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{7}}
      .with_variant_options(legate::VariantOptions{}.with_may_throw_exception(true))
      .with_signature(legate::TaskSignature{}.inputs(1).scalars(2));

  static void cpu_variant(legate::TaskContext context);
};
