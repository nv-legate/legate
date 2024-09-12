/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/utilities/compiler.h"

#include <cstdlib>
#include <cxxabi.h>
#include <iomanip>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>

namespace legate::detail {

namespace {

template <typename... T>
[[nodiscard]] std::string make_error(std::string_view mangled_name,
                                     std::string_view mess,
                                     T&&... rest)
{
  std::stringstream ss;

  ss << "error demangling " << std::quoted(mangled_name) << ": " << mess;
  ((ss << ' ' << std::forward<T>(rest)), ...);
  return std::move(ss).str();
}

}  // namespace

std::string demangle_type(const std::type_info& ti)
{
  const auto* mangled_name = ti.name();
  int status               = 0;
  const auto demangled     = std::unique_ptr<char, void (*)(void*)>{
    abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status), std::free};

  // See https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
  switch (status) {
    case 0: break;  // no error
    case -1: throw std::bad_alloc{};
    case -2: {
      const auto ss =
        make_error(mangled_name, "it is not a valid name under the C++ ABI mangling rules");
      throw std::domain_error{ss};
    }
    case -3: {
      const auto ss = make_error(mangled_name, "invalid arguments passed to abi::__cxa_demangle()");
      throw std::invalid_argument{ss};
    }
    default: {
      const auto ss = make_error(
        mangled_name, "unknown failure calling abi::__cxa_demangle(), error code:", status);
      throw std::runtime_error{ss};
    }
  }
  return demangled.get();
}

}  // namespace legate::detail
