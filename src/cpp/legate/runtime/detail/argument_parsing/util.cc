/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/util.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/base.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace legate::detail {

template <typename StringType>
std::vector<StringType> string_split(std::string_view command, const char sep)
{
  std::vector<StringType> qargs;

  LEGATE_ASSERT(sep != '\"' && sep != '\'');
  while (!command.empty()) {
    std::size_t arglen;
    auto quoted = false;

    if (const auto c = command.front(); c == '\"' || c == '\'') {
      command.remove_prefix(1);
      quoted = true;
      arglen = command.find(c);
      if (arglen == std::string_view::npos) {
        throw TracedException<std::invalid_argument>{
          fmt::format("Unterminated quote: '{}'", command)};
      }
    } else if (c == sep) {
      command.remove_prefix(1);
      continue;
    } else {
      arglen = std::min(command.find(sep), command.size());
    }

    if (const auto sub = command.substr(0, arglen); !sub.empty()) {
      qargs.emplace_back(sub);
    }
    command.remove_prefix(arglen + quoted);
  }
  return qargs;
}

#ifndef DOXYGEN
template std::vector<std::string> string_split(std::string_view command, const char sep);
template std::vector<std::string_view> string_split(std::string_view command, const char sep);
#endif

bool multi_node_job()
{
  constexpr EnvironmentVariable<std::uint32_t> OMPI_COMM_WORLD_SIZE{"OMPI_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> MV2_COMM_WORLD_SIZE{"MV2_COMM_WORLD_SIZE"};
  constexpr EnvironmentVariable<std::uint32_t> SLURM_NTASKS{"SLURM_NTASKS"};

  return OMPI_COMM_WORLD_SIZE.get().value_or(1) > 1 ||  //
         MV2_COMM_WORLD_SIZE.get().value_or(1) > 1 ||   //
         SLURM_NTASKS.get().value_or(1) > 1;
}

std::vector<std::string> deduplicate_command_line_flags(Span<const std::string> args)
{
  // A dummy name that is used only in case the first arguments are positional. Currently
  // LEGATE_CONFIG does not actually have any such arguments, but good to be forward-looking.
  auto arg_name = std::string_view{"==POSITIONAL=FIRST=ARGUMENTS=="};
  // We want to order the flags in the *reverse* order in which they are found, so given:
  //
  // --foo --bar --foo
  //
  // we want to emit
  //
  // --bar --foo
  //
  // The rationale here is that ordering *probably* makes no difference, but we cannot be sure,
  // so best to preserve it.
  auto reverse_flag_order = std::vector<std::string_view>{arg_name};
  auto values             = std::unordered_map<std::string_view, std::vector<std::string_view>>{};

  reverse_flag_order.reserve(args.size());
  values.reserve(args.size());
  for (auto&& arg : args) {
    if (arg.find('-') == 0) {
      arg_name = std::string_view{arg}.substr(0, arg.find('='));

      if (const auto [values_it, inserted] = values.try_emplace(arg_name); inserted) {
        reverse_flag_order.push_back(arg_name);
      } else {
        // We have seen the flag before, so we need to:
        //
        // 1. Clear the previous values.
        values_it->second.clear();
        // 2. Reorder the flag order to move it to the back. We use std::rotate() here instead
        //    of swap because we want to preserve the relative order of the flags in between as
        //    well.
        auto it = std::find(reverse_flag_order.begin(), reverse_flag_order.end(), arg_name);

        std::rotate(it, std::next(it), reverse_flag_order.end());
      }
    }
    values[arg_name].push_back(arg);
  }

  LEGATE_CHECK(reverse_flag_order.size() == values.size());

  auto ret = std::vector<std::string>{};

  // This will over-reserve since it doesn't take into account any de-duplication.
  ret.reserve(args.size());
  for (auto&& f : reverse_flag_order) {
    auto vit = values.find(f);

    LEGATE_CHECK(vit != values.end());
    ret.insert(
      ret.end(), std::move_iterator{vit->second.begin()}, std::move_iterator{vit->second.end()});
    values.erase(vit);
  }
  LEGATE_CHECK(values.empty());
  return ret;
}

}  // namespace legate::detail
