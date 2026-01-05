/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/formatters.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/mapping/detail/machine.h>
#include <legate/mapping/mapping.h>
#include <legate/operation/detail/operation.h>
#include <legate/operation/detail/task.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/task/detail/task_info.h>
#include <legate/task/detail/variant_info.h>
#include <legate/task/task_info.h>
#include <legate/type/detail/types.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

namespace fmt {

format_context::iterator formatter<legate::detail::Type>::format(const legate::detail::Type& a,
                                                                 format_context& ctx) const
{
  return formatter<std::string>::format(a.to_string(), ctx);
}

format_context::iterator formatter<legate::Type>::format(const legate::Type& a,
                                                         format_context& ctx) const
{
  return formatter<std::string>::format(a.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Operation>::format(
  const legate::detail::Operation& op, format_context& ctx) const
{
  return formatter<std::string>::format(op.to_string(true /*show_provenance*/), ctx);
}

format_context::iterator formatter<legate::detail::Shape>::format(
  const legate::detail::Shape& shape, format_context& ctx) const
{
  return formatter<std::string>::format(shape.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Constraint>::format(
  const legate::detail::Constraint& constraint, format_context& ctx) const
{
  return formatter<std::string>::format(constraint.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Variable>::format(
  const legate::detail::Variable& var, format_context& ctx) const
{
  return formatter<std::string>::format(var.to_string(), ctx);
}

format_context::iterator formatter<legate::VariantCode>::format(legate::VariantCode variant,
                                                                format_context& ctx) const
{
  std::string_view name = "(unknown)";

  switch (variant) {
#define LEGATE_VARIANT_CASE(x) \
  case legate::VariantCode::x: name = #x "_VARIANT"; break
    LEGATE_VARIANT_CASE(CPU);
    LEGATE_VARIANT_CASE(GPU);
    LEGATE_VARIANT_CASE(OMP);
#undef LEGATE_VARIANT_CASE
  }

  return formatter<std::string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::LocalTaskID>::format(legate::LocalTaskID id,
                                                                format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::LocalTaskID>>::format(fmt::underlying(id), ctx);
}

format_context::iterator formatter<legate::GlobalTaskID>::format(legate::GlobalTaskID id,
                                                                 format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::GlobalTaskID>>::format(fmt::underlying(id), ctx);
}

format_context::iterator formatter<legate::LocalRedopID>::format(legate::LocalRedopID id,
                                                                 format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::LocalRedopID>>::format(fmt::underlying(id), ctx);
}

format_context::iterator formatter<legate::GlobalRedopID>::format(legate::GlobalRedopID id,
                                                                  format_context& ctx) const
{
  return formatter<std::underlying_type_t<legate::GlobalRedopID>>::format(fmt::underlying(id), ctx);
}

format_context::iterator formatter<legate::ImageComputationHint>::format(
  legate::ImageComputationHint hint, format_context& ctx) const
{
  std::string_view name = "(unknown)";
  switch (hint) {
#define LEGATE_HINT_CASE(x) \
  case legate::ImageComputationHint::x: name = LEGATE_STRINGIZE_(x); break
    LEGATE_HINT_CASE(NO_HINT);
    LEGATE_HINT_CASE(MIN_MAX);
    LEGATE_HINT_CASE(FIRST_LAST);
#undef LEGATE_HINT_CASE
  };

  return formatter<std::string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::detail::ZStringView>::format(
  const legate::detail::ZStringView& sv, format_context& ctx) const
{
  return formatter<std::string_view>::format(sv.as_string_view(), ctx);
}

format_context::iterator formatter<legate::detail::LogicalRegionField>::format(
  const legate::detail::LogicalRegionField& lrf, format_context& ctx) const
{
  return formatter<std::string>::format(lrf.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::Storage>::format(
  const legate::detail::Storage& s, format_context& ctx) const
{
  return formatter<std::string>::format(s.to_string(), ctx);
}

format_context::iterator formatter<legate::TaskInfo>::format(const legate::TaskInfo& info,
                                                             format_context& ctx) const
{
  return formatter<std::string>::format(info.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::TaskInfo>::format(
  const legate::detail::TaskInfo& info, format_context& ctx) const
{
  return formatter<std::string>::format(info.to_string(), ctx);
}

format_context::iterator formatter<legate::detail::VariantInfo>::format(
  const legate::detail::VariantInfo& info, format_context& ctx) const
{
  return formatter<std::string>::format(info.to_string(), ctx);
}

format_context::iterator formatter<legate::mapping::detail::Machine>::format(
  const legate::mapping::detail::Machine& machine, format_context& ctx) const
{
  return formatter<std::string>::format(machine.to_string(), ctx);
}

format_context::iterator formatter<legate::mapping::TaskTarget>::format(
  legate::mapping::TaskTarget target, format_context& ctx) const
{
  std::string_view name = "(unknown)";
  switch (target) {
#define LEGATE_TASK_TARGET_CASE(x) \
  case legate::mapping::TaskTarget::x: name = #x; break
    LEGATE_TASK_TARGET_CASE(GPU);
    LEGATE_TASK_TARGET_CASE(CPU);
    LEGATE_TASK_TARGET_CASE(OMP);
#undef LEGATE_TASK_TARGET_CASE
  }
  return formatter<std::string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::mapping::StoreTarget>::format(
  legate::mapping::StoreTarget target, format_context& ctx) const
{
  std::string_view name = "(unknown)";
  switch (target) {
#define LEGATE_STORE_TARGET_CASE(x) \
  case legate::mapping::StoreTarget::x: name = #x; break
    LEGATE_STORE_TARGET_CASE(FBMEM);
    LEGATE_STORE_TARGET_CASE(ZCMEM);
    LEGATE_STORE_TARGET_CASE(SYSMEM);
    LEGATE_STORE_TARGET_CASE(SOCKETMEM);
#undef LEGATE_STORE_TARGET_CASE
  }
  return formatter<std::string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::detail::LogicalTask>::format(
  const legate::detail::LogicalTask& task, format_context& ctx) const
{
  return formatter<std::string>::format(task.to_string(/* show provenance */ false), ctx);
}

format_context::iterator formatter<legate::detail::LogicalStore>::format(
  const legate::detail::LogicalStore& store, format_context& ctx) const
{
  return formatter<std::string>::format(store.to_string(), ctx);
}

format_context::iterator formatter<legion_privilege_mode_t>::format(legion_privilege_mode_t mode,
                                                                    format_context& ctx) const
{
  std::string_view name = "(unknown)";
  switch (mode) {
#define LEGION_PRIVILEGE_MODE_CASE(x) \
  case x: name = #x; break
    LEGION_PRIVILEGE_MODE_CASE(LEGION_READ_ONLY);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_READ_DISCARD);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_REDUCE);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_WRITE_ONLY);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_READ_WRITE);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_WRITE_DISCARD);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_WRITE_PRIV);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_NO_ACCESS);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_DISCARD_MASK);
    LEGION_PRIVILEGE_MODE_CASE(LEGION_DISCARD_OUTPUT_MASK);
#undef LEGION_PRIVILEGE_MODE_CASE
  }
  return formatter<std::string_view>::format(name, ctx);
}

format_context::iterator formatter<legate::StreamingMode>::format(legate::StreamingMode mode,
                                                                  format_context& ctx) const
{
  std::string_view name = "(unknown)";
  switch (mode) {
#define LEGATE_STREAMING_MODE_CASE(x) \
  case legate::StreamingMode::x: name = #x; break
    LEGATE_STREAMING_MODE_CASE(OFF);
    LEGATE_STREAMING_MODE_CASE(RELAXED);
    LEGATE_STREAMING_MODE_CASE(STRICT);
#undef LEGATE_STREAMING_MODE_CASE
  }
  return formatter<std::string_view>::format(name, ctx);
}

}  // namespace fmt
