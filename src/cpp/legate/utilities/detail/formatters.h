/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <string>
#include <type_traits>

namespace legate {

class Point;
class Rect;

enum class LocalTaskID : std::int64_t;
enum class VariantCode : unsigned int /* A.K.A. Legion::VariantID */;

enum class LocalRedopID : std::int64_t;
enum class GlobalRedopID : int /* A.K.A. Legion::ReductionOpID */;

enum class ImageComputationHint : std::uint8_t;

enum class StreamingMode : std::uint8_t;

class Type;
class TaskInfo;

}  // namespace legate

namespace legate::detail {

class Type;
class Operation;
class Shape;
class Constraint;
class Variable;
class LogicalRegionField;
class Storage;
class VariantInfo;
class TaskInfo;
class LogicalTask;
class LogicalStore;

template <typename CharT, typename TraitsT>
class BasicZStringView;

using ZStringView = BasicZStringView<char, std::char_traits<char>>;

}  // namespace legate::detail

namespace legate::mapping {

enum class TaskTarget : std::uint8_t;
enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate::mapping::detail {

class Machine;

}  // namespace legate::mapping::detail

namespace fmt {

template <>
struct formatter<legate::detail::Type> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Type& a, format_context& ctx) const;
};

template <>
struct formatter<legate::Type> : formatter<std::string> {
  format_context::iterator format(const legate::Type& a, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Operation> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Operation& op, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Shape> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Shape& shape, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Constraint> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Constraint& constraint,
                                  format_context& ctx) const;
};

template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_base_of_v<legate::detail::Constraint, T>>>
  : formatter<legate::detail::Constraint, Char> {
  format_context::iterator format(const T& constraint, format_context& ctx) const
  {
    return formatter<legate::detail::Constraint, Char>::format(constraint, ctx);
  }
};

template <>
struct formatter<legate::detail::Variable> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Variable& var, format_context& ctx) const;
};

template <>
struct formatter<legate::VariantCode> : formatter<std::string_view> {
  format_context::iterator format(legate::VariantCode variant, format_context& ctx) const;
};

template <>
struct formatter<legate::LocalTaskID> : formatter<std::underlying_type_t<legate::LocalTaskID>> {
  format_context::iterator format(legate::LocalTaskID id, format_context& ctx) const;
};

template <>
struct formatter<legate::LocalRedopID> : formatter<std::underlying_type_t<legate::LocalRedopID>> {
  format_context::iterator format(legate::LocalRedopID id, format_context& ctx) const;
};

template <>
struct formatter<legate::GlobalRedopID> : formatter<std::underlying_type_t<legate::GlobalRedopID>> {
  format_context::iterator format(legate::GlobalRedopID id, format_context& ctx) const;
};

template <>
struct formatter<legate::ImageComputationHint> : formatter<std::string_view> {
  format_context::iterator format(legate::ImageComputationHint hint, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::ZStringView> : formatter<std::string_view> {
  format_context::iterator format(const legate::detail::ZStringView& sv, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::LogicalRegionField> : formatter<std::string> {
  format_context::iterator format(const legate::detail::LogicalRegionField& lrf,
                                  format_context& ctx) const;
};

template <>
struct formatter<legate::detail::Storage> : formatter<std::string> {
  format_context::iterator format(const legate::detail::Storage& s, format_context& ctx) const;
};

template <>
struct formatter<legate::TaskInfo> : formatter<std::string> {
  format_context::iterator format(const legate::TaskInfo& info, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::TaskInfo> : formatter<std::string> {
  format_context::iterator format(const legate::detail::TaskInfo& info, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::VariantInfo> : formatter<std::string> {
  format_context::iterator format(const legate::detail::VariantInfo& info,
                                  format_context& ctx) const;
};

template <>
struct formatter<legate::mapping::detail::Machine> : formatter<std::string> {
  format_context::iterator format(const legate::mapping::detail::Machine& machine,
                                  format_context& ctx) const;
};

template <>
struct formatter<legate::mapping::TaskTarget> : formatter<std::string_view> {
  format_context::iterator format(legate::mapping::TaskTarget, format_context& ctx) const;
};

template <>
struct formatter<legate::mapping::StoreTarget> : formatter<std::string_view> {
  format_context::iterator format(legate::mapping::StoreTarget target, format_context& ctx) const;
};

template <>
struct formatter<legate::detail::LogicalTask> : formatter<std::string> {
  format_context::iterator format(const legate::detail::LogicalTask& task,
                                  format_context& ctx) const;
};

template <typename T, typename Char>
struct formatter<T, Char, std::enable_if_t<std::is_base_of_v<legate::detail::LogicalTask, T>>>
  : formatter<legate::detail::LogicalTask, Char> {
  format_context::iterator format(const T& task, format_context& ctx) const
  {
    return formatter<legate::detail::LogicalTask, Char>::format(task, ctx);
  }
};

template <>
struct formatter<legate::detail::LogicalStore> : formatter<std::string> {
  format_context::iterator format(const legate::detail::LogicalStore& store,
                                  format_context& ctx) const;
};

template <>
struct formatter<legate::Point> : ostream_formatter {};

template <>
struct formatter<legate::Rect> : ostream_formatter {};

template <>
struct formatter<legate::StreamingMode> : formatter<std::string_view> {
  format_context::iterator format(legate::StreamingMode mode, format_context& ctx) const;
};

}  // namespace fmt
