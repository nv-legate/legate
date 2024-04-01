/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/runtime/scope.h"

#include "core/runtime/detail/runtime.h"

#include <optional>
#include <utility>

namespace legate {

class Scope::Impl {
 public:
  void set_priority(std::int32_t priority)
  {
    if (priority_) {
      throw std::invalid_argument{"Priority can be set only once for each scope"};
    }
    priority_ = detail::Runtime::get_runtime()->scope().exchange_priority(priority);
  }

  void set_exception_mode(ExceptionMode exception_mode)
  {
    if (exception_mode_) {
      throw std::invalid_argument{"Exception mode can be set only once for each scope"};
    }
    exception_mode_ =
      detail::Runtime::get_runtime()->scope().exchange_exception_mode(exception_mode);
  }

  void set_provenance(std::string provenance)
  {
    if (provenance_) {
      throw std::invalid_argument{"Provenance can be set only once for each scope"};
    }
    provenance_ =
      detail::Runtime::get_runtime()->scope().exchange_provenance(std::move(provenance));
  }

  void set_machine(InternalSharedPtr<mapping::detail::Machine> machine)
  {
    if (machine_) {
      throw std::invalid_argument{"Machine can be set only once for each scope"};
    }
    machine_ = detail::Runtime::get_runtime()->scope().exchange_machine(std::move(machine));
  }

  ~Impl()
  {
    if (priority_) {
      static_cast<void>(detail::Runtime::get_runtime()->scope().exchange_priority(*priority_));
    }
    if (exception_mode_) {
      static_cast<void>(
        detail::Runtime::get_runtime()->scope().exchange_exception_mode(*exception_mode_));
    }
    if (provenance_) {
      static_cast<void>(
        detail::Runtime::get_runtime()->scope().exchange_provenance(std::move(*provenance_)));
    }
    if (machine_) {
      static_cast<void>(
        detail::Runtime::get_runtime()->scope().exchange_machine(std::move(machine_)));
    }
  }

 private:
  std::optional<std::int32_t> priority_{};
  std::optional<ExceptionMode> exception_mode_{};
  std::optional<std::string> provenance_{};
  InternalSharedPtr<mapping::detail::Machine> machine_{};
};

template class default_delete<Scope::Impl>;

Scope::Scope() : impl_{new Scope::Impl{}} {}

Scope::Scope(std::int32_t priority) : Scope{} { set_priority(priority); }

Scope::Scope(ExceptionMode exception_mode) : Scope{} { set_exception_mode(exception_mode); }

Scope::Scope(std::string provenance) : Scope{} { set_provenance(std::move(provenance)); }

Scope::Scope(const mapping::Machine& machine) : Scope{} { set_machine(machine); }

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
    throw std::runtime_error{"Empty machines cannot be used for resource scoping"};
  }
  impl_->set_machine(result.impl());
}

Scope::~Scope() = default;

/*static*/ std::int32_t Scope::priority()
{
  return detail::Runtime::get_runtime()->scope().priority();
}

/*static*/ legate::ExceptionMode Scope::exception_mode()
{
  return detail::Runtime::get_runtime()->scope().exception_mode();
}

/*static*/ const std::string& Scope::provenance()
{
  return detail::Runtime::get_runtime()->scope().provenance();
}

/*static*/ mapping::Machine Scope::machine()
{
  return mapping::Machine{detail::Runtime::get_runtime()->scope().machine()};
}

}  // namespace legate
