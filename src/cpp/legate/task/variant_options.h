/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/cpp_version.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/typedefs.h>

#include <array>
#include <cstddef>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file
 * @brief Class definition of legate::VariantOptions
 */
namespace legate {

class VariantOptions;

namespace cython_detail {

void set_new_comms(std::vector<std::string> comms, legate::VariantOptions* options);

}  // namespace cython_detail

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A helper class for specifying variant options
 */
class VariantOptions {
 public:
  /**
   * @brief Whether the variant needs a concurrent task launch. `false` by default.
   *
   * Normally, leaf tasks (i.e. all individual task instances created by a single launch) are
   * allowed to execute in any order so long as their preconditions are met. For example, if a
   * task is launched that creates 100 leaf tasks, those tasks can execute at any time so long
   * as each individual task's inputs are satisfied. It is even possible to have other leaf
   * tasks (from other tasks) executing at the same time or between them.
   *
   * Setting `concurrent` to `true` says: if this task is parallelized, then all leaf tasks
   * must execute concurrently. Note, concurrency is a *requirement*, not a grant. The entire
   * machine must execute the tasks at exactly the same time as one giant block. No other tasks
   * marked concurrent may execute at the same time.
   *
   * Setting `concurrent` to `false` (the default) says: the task can execute as normal. The
   * leaf tasks can execute in any order.
   *
   * This feature is most often used when doing collective communications (i.e. all-reduce,
   * all-gather) inside the tasks. In this case, the tasks need to execute in lockstep because
   * otherwise deadlocks may occur.
   *
   * Suppose there are 2 tasks (A and B) that do collectives. If they execute without
   * concurrency, it is possible for half of the "task A" tasks and half of the "task B" tasks
   * to be running at the same time. Eventually each of those tasks will reach a point where
   * they must all-gather. The program would deadlock because both sides would be waiting for
   * the communication that would never be able to finish.
   *
   * For this reason, adding any communicators (see `communicators`) automatically implies
   * `concurrent = true`.
   */
  bool concurrent{false};

  /**
   * @brief If the flag is `true`, the variant is allowed to create buffers (temporary or output)
   * during execution. `false` by default.
   */
  bool has_allocations{false};

  /**
   * @brief Whether this variant can skip device context synchronization after completion.
   *
   * Normally, for device-enabled task variants, Legate will emit a device-wide barrier to
   * ensure that all outstanding (potentially asynchronous) work performed by the variant has
   * completed. However, if the task launches no such work, or if that work is launched using
   * the task-specific device streams, then such a context synchronization is not necessary.
   *
   * Setting this value to `true` ensures that no context synchronization is performed. Setting
   * it to `false` guarantees that a context synchronization is done.
   *
   * Has no effect on non-device variants (for example CPU variants).
   *
   * @see with_elide_device_ctx_sync()
   */
  bool elide_device_ctx_sync{};

  /**
   * @brief Indicate whether a task has side effects outside of the runtime's tracking that
   * forbid it from replicated a task.
   *
   * When a task only takes scalar stores, it gets replicated by default on all the ranks, as
   * that's more efficient than having only one of the ranks run it and broadcast the results.
   *
   * However, sometimes a task may have "side effects" (which are outside the runtime's
   * tracking) which should otherwise forbid the runtime from replicating a particular variant.
   *
   * For example, the task may write something to disk, or effect some other kind of permanent
   * change to the system. In these cases the runtime must not replicate the task, as the
   * effect must occur exactly once.
   */
  bool has_side_effect{};

  /**
   * @brief Whether this variant may throw an exception.
   *
   * Tasks that throw exception must be handled specially by the runtime in order to safely and
   * correctly propagate the thrown exceptions. For this reason, tasks must explicitly declare
   * whether they throw an exception.
   *
   * @warning This special handling usually comes with severe performance penalties. For
   * example, the runtime may block the calling thread (i.e. the main thread) on the completion
   * of the possibly throwing task, or may opt not to schedule any other tasks concurrently.
   *
   * @warning It is highly recommended that tasks do *not* throw exceptions, and instead
   * indicate an error state using some other way. Exceptions should be used as an absolute
   * last resort.
   */
  bool may_throw_exception{};

  /**
   * @brief The maximum number of communicators allowed per variant.
   *
   * This is a workaround for insufficient constexpr support in C++17 and will be removed in a
   * future release.
   */
  static constexpr auto MAX_COMMS = 3;

  /**
   * @brief The communicator(s) to be used by the variant, or `std::nullopt` if no communicator
   * is to be used.
   *
   * Setting this to anything other than `std::nullopt` implies `concurrent` to be `true`.
   */
  std::optional<std::array<std::string_view, MAX_COMMS>> communicators{};

  LEGATE_CPP_VERSION_TODO(20, "Use std::vector for underlying container, and get rid of MAX_COMMS");

  /**
   * @brief Changes the value of the `concurrent` flag
   *
   * @param `concurrent` A new value for the `concurrent` flag
   */
  constexpr VariantOptions& with_concurrent(bool concurrent);
  /**
   * @brief Changes the value of the `has_allocations` flag
   *
   * @param `has_allocations` A new value for the `has_allocations` flag
   */
  constexpr VariantOptions& with_has_allocations(bool has_allocations);

  /**
   * @brief Sets whether the variant can elide device context synchronization after task
   * completion.
   *
   * @param `elide_sync` `true` if this variant can skip synchronizing the device context after
   * task completion, `false` otherwise.
   *
   * @return reference to `this`.
   *
   * @see elide_device_ctx_sync
   */
  constexpr VariantOptions& with_elide_device_ctx_sync(bool elide_sync);

  /**
   * @brief Sets whether the variant has side effects.
   *
   * @param side_effect `true` if the task has side-effects, `false` otherwise.
   *
   * @return reference to `this`.
   *
   * @see has_side_effect.
   */
  constexpr VariantOptions& with_has_side_effect(bool side_effect);

  /**
   * @brief Sets whether the variant may throw exceptions.
   *
   * @param may_throw `true` if the variant may throw exceptions, `false` otherwise.
   *
   * @return reference to `this`.
   *
   * @see may_throw_exception.
   */
  constexpr VariantOptions& with_may_throw_exception(bool may_throw);

  /**
   * @brief Sets the communicator(s) for the variant.
   *
   * This call implies `concurrent = true` as well.
   *
   * The `VariantOptions` does not take ownership of `comms` in any way. If `comms` are not
   * constructed from a string-literal, or some other object with static storage duration, then
   * the user must ensure that the string(s) outlives this object.
   *
   * Due to limitations with constexpr in C++17, the user may register at most `MAX_COMMS`
   * number of communicators. This restriction is expected to be lifted in the future.
   *
   * @param comms The communicator(s) to use.
   *
   * @return reference to `this`.
   *
   * @see communicators.
   */
  VariantOptions& with_communicators(std::initializer_list<std::string_view> comms) noexcept;

  LEGATE_CPP_VERSION_TODO(20, "The above function can be constexpr");

  class WithCommunicatorsAccessKey {
    WithCommunicatorsAccessKey() = default;

    friend class VariantOptions;
    friend void cython_detail::set_new_comms(std::vector<std::string>, legate::VariantOptions*);
  };

  template <typename It>
  VariantOptions& with_communicators(WithCommunicatorsAccessKey, It begin, It end) noexcept;

  /**
   * @brief Populate a Legion::TaskVariantRegistrar using the options contained.
   *
   * @param registrar The registrar to fill out.
   */
  void populate_registrar(Legion::TaskVariantRegistrar& registrar) const;

  [[nodiscard]] constexpr bool operator==(const VariantOptions& other) const;
  [[nodiscard]] constexpr bool operator!=(const VariantOptions& other) const;

  /**
   * @brief The default variant options used during task creation if no user-supplied options
   * are given.
   */
  static const VariantOptions DEFAULT_OPTIONS;
};

// This trick is needed because you cannot declare a constexpr variable of the same class
// inside the class definition, because at that point the class is still considered an
// incomplete type.
//
// Do not be fooled, DEFAULT_VARIANT_OPTIONS is still constexpr; for variables, constexpr can
// explicitly only be on a definition, not on any declarations
// (eel.is/c++draft/dcl.constexpr#1.sentence-1). The static const is the declaration, the line
// below is the definition.
inline constexpr VariantOptions VariantOptions::DEFAULT_OPTIONS{};

std::ostream& operator<<(std::ostream& os, const VariantOptions& options);

/** @} */

}  // namespace legate

#include <legate/task/variant_options.inl>
