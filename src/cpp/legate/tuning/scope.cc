/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/tuning/scope.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <optional>
#include <stdexcept>
#include <utility>

namespace legate {

class Scope::Impl {
 public:
  void set_priority(std::int32_t priority)
  {
    if (priority_) {
      throw detail::TracedException<std::invalid_argument>{
        "Priority can be set only once for each scope"};
    }
    priority_ = detail::Runtime::get_runtime().scope().exchange_priority(priority);
  }

  void set_exception_mode(ExceptionMode exception_mode)
  {
    if (exception_mode_) {
      throw detail::TracedException<std::invalid_argument>{
        "Exception mode can be set only once for each scope"};
    }
    exception_mode_ =
      detail::Runtime::get_runtime().scope().exchange_exception_mode(exception_mode);
  }

  void set_provenance(std::string provenance)
  {
    if (provenance_) {
      throw detail::TracedException<std::invalid_argument>{
        "Provenance can be set only once for each scope"};
    }
    provenance_ = detail::Runtime::get_runtime().scope().exchange_provenance(std::move(provenance));
  }

  void set_machine(InternalSharedPtr<mapping::detail::Machine> machine)
  {
    if (machine_) {
      throw detail::TracedException<std::invalid_argument>{
        "Machine can be set only once for each scope"};
    }
    machine_ = detail::Runtime::get_runtime().scope().exchange_machine(std::move(machine));
  }

  void set_parallel_policy(ParallelPolicy parallel_policy)
  {
    if (parallel_policy_) {
      throw detail::TracedException<std::invalid_argument>{
        "Parallel policy can be set only once for each scope"};
    }

    const auto new_is_streaming = parallel_policy.streaming();
    auto& global_scope          = detail::Runtime::get_runtime().scope();

    // If the window size isn't big enough you never get any streaming, because every streaming
    // run will be size 1, and so will just get mapped normally. So we want to artificially
    // increase the scheduling window size (to some suitable large window) for the duration of
    // a streaming scope.
    //
    // This scope variable is not yet fully exposed to the user (they are only allowed to
    // toggle it globally, at program startup), but in case it ever is exposed, respect their
    // scheduling window value first.
    if (!scheduling_window_size_.has_value()) {
      constexpr auto BIG_WINDOW = 1024U;

      if (new_is_streaming && (global_scope.scheduling_window_size() < BIG_WINDOW)) {
        scheduling_window_size_ = global_scope.exchange_scheduling_window_size(BIG_WINDOW);
      }
    }

    // this should be the last action in case the control returns earlier due to an
    // exception begin thrown
    try {
      global_scope.trigger_exchange_side_effects(parallel_policy,
                                                 detail::Scope::ChangeKind::SCOPE_BEG);
      parallel_policy_ = global_scope.exchange_parallel_policy(std::move(parallel_policy));

    } catch (const std::exception& e) {
      parallel_policy_ = global_scope.exchange_parallel_policy(std::move(parallel_policy));
      throw;
    }
  }

  /**
   * @brief Initiate actions required when scope is about to end.
   *
   * When the Scope ends and the ParallelPolicy is replaced, it may trigger
   * additional actions, such as flushing the scheduling window.
   *
   * @throws std::invalid_argument exception if operations submitted in the scope
   * are not streamable.
   */
  void trigger_end_scope()
  {
    if (parallel_policy_.has_value()) {
      auto& global_scope = detail::Runtime::get_runtime().scope();

      global_scope.trigger_exchange_side_effects(*parallel_policy_,
                                                 detail::Scope::ChangeKind::SCOPE_END);
    }
  }

  /**
   * @brief Separate out the destructor actions as a separate function so that we
   * can call them inside a catch clause inside Scope's destructor, in order to
   * restore the global scope to correct state.
   */
  void destruct()
  {
    auto& global_scope = detail::Runtime::get_runtime().scope();

    if (priority_.has_value()) {
      static_cast<void>(global_scope.exchange_priority(*priority_));
      priority_ = std::nullopt;
    }
    if (exception_mode_.has_value()) {
      static_cast<void>(global_scope.exchange_exception_mode(*exception_mode_));
      exception_mode_ = std::nullopt;
    }
    if (provenance_.has_value()) {
      static_cast<void>(global_scope.exchange_provenance(std::move(*provenance_)));
      provenance_ = std::nullopt;
    }
    if (machine_) {
      static_cast<void>(global_scope.exchange_machine(std::move(machine_)));
      machine_ = nullptr;
    }
    if (parallel_policy_.has_value()) {
      static_cast<void>(global_scope.exchange_parallel_policy(std::move(*parallel_policy_)));
      parallel_policy_ = std::nullopt;
    }
    if (scheduling_window_size_.has_value()) {
      static_cast<void>(global_scope.exchange_scheduling_window_size(*scheduling_window_size_));
      scheduling_window_size_ = std::nullopt;
    }
  }

  ~Impl() { destruct(); }

 private:
  std::optional<std::int32_t> priority_{};
  std::optional<ExceptionMode> exception_mode_{};
  std::optional<std::string> provenance_{};
  InternalSharedPtr<mapping::detail::Machine> machine_{};
  std::optional<ParallelPolicy> parallel_policy_{};
  std::optional<std::uint32_t> scheduling_window_size_{};
};

template class DefaultDelete<Scope::Impl>;

Scope::Scope() : impl_{new Scope::Impl{}} {}

Scope::Scope(std::int32_t priority) : Scope{} { set_priority(priority); }

Scope::Scope(ExceptionMode exception_mode) : Scope{} { set_exception_mode(exception_mode); }

Scope::Scope(std::string provenance) : Scope{} { set_provenance(std::move(provenance)); }

Scope::Scope(const mapping::Machine& machine) : Scope{} { set_machine(machine); }

Scope::Scope(ParallelPolicy parallel_policy) : Scope{}
{
  set_parallel_policy(std::move(parallel_policy));
}

Scope&& Scope::with_priority(std::int32_t priority) &&
{
  set_priority(priority);
  return std::move(*this);
}

Scope&& Scope::with_exception_mode(ExceptionMode exception_mode) &&
{
  set_exception_mode(exception_mode);
  return std::move(*this);
}

Scope&& Scope::with_provenance(std::string provenance) &&
{
  set_provenance(std::move(provenance));
  return std::move(*this);
}

Scope&& Scope::with_machine(const mapping::Machine& machine) &&
{
  set_machine(machine);
  return std::move(*this);
}

Scope&& Scope::with_parallel_policy(ParallelPolicy parallel_policy) &&
{
  set_parallel_policy(std::move(parallel_policy));
  return std::move(*this);
}

void Scope::set_priority(std::int32_t priority) { impl_->set_priority(priority); }

void Scope::set_exception_mode(ExceptionMode exception_mode)
{
  impl_->set_exception_mode(exception_mode);
}

void Scope::set_provenance(std::string provenance) { impl_->set_provenance(std::move(provenance)); }

void Scope::set_machine(const mapping::Machine& machine)
{
  auto result = Scope::machine() & machine;
  if (result.empty()) {
    throw detail::TracedException<std::runtime_error>{
      "Empty machines cannot be used for resource scoping"};
  }
  impl_->set_machine(result.impl());
}

void Scope::set_parallel_policy(ParallelPolicy parallel_policy)
{
  impl_->set_parallel_policy(std::move(parallel_policy));
}

Scope::~Scope() noexcept(false)
{
  if (impl_) {
    try {
      impl_->trigger_end_scope();
    } catch (const std::exception& e) {
      // must cleanup and restore the global scope correctly.
      impl_->destruct();
      throw;
    }
  }
}

/*static*/ std::int32_t Scope::priority()
{
  return detail::Runtime::get_runtime().scope().priority();
}

/*static*/ legate::ExceptionMode Scope::exception_mode()
{
  return detail::Runtime::get_runtime().scope().exception_mode();
}

/*static*/ std::string_view Scope::provenance()
{
  return detail::Runtime::get_runtime().scope().provenance().as_string_view();
}

/*static*/ mapping::Machine Scope::machine()
{
  return mapping::Machine{detail::Runtime::get_runtime().scope().machine()};
}

/*static*/ const ParallelPolicy& Scope::parallel_policy()
{
  return detail::Runtime::get_runtime().scope().parallel_policy();
}

}  // namespace legate
