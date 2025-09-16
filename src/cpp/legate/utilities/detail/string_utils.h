/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace legate::detail {

/**
 * @brief Convert all characters in a string to lowercase.
 *
 * @param s Input string.
 *
 * @return A copy of the input string with all characters converted to lowercase.
 */
[[nodiscard]] std::string string_to_lower(std::string s);

/**
 * @brief Remove leading whitespace characters from a string.
 *
 * @param s Input string.
 *
 * @return A copy of the input string with leading whitespace removed.
 */
[[nodiscard]] std::string string_lstrip(std::string s);

/**
 * @brief Remove trailing whitespace characters from a string.
 *
 * @param s Input string.
 *
 * @return A copy of the input string with trailing whitespace removed.
 */
[[nodiscard]] std::string string_rstrip(std::string s);

/**
 * @brief Remove leading and trailing whitespace characters from a string.
 *
 * @param s Input string.
 *
 * @return A copy of the input string with leading and trailing whitespace removed.
 */
[[nodiscard]] std::string string_strip(std::string s);

/**
 * @brief Split a string into substrings using a specified delimiter.
 *
 * `sep` must not be `'` or `"`.
 *
 * @tparam StringType Type of string to store each split part (defaults to std::string).
 *
 * @param sv Input string view to split.
 * @param sep Delimiter character used for splitting (defaults to space).
 *
 * @return A vector containing the split substrings.
 *
 * @throw std::invalid_argument If `sv` contains an unterminated quote.
 */
template <typename StringType = std::string>
[[nodiscard]] std::vector<StringType> string_split(std::string_view sv, char sep = ' ');

}  // namespace legate::detail
