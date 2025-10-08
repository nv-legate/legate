/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/shared_library.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <dlfcn.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace legate::detail {

namespace {

[[nodiscard]] std::unique_ptr<void, int (*)(void*)> load_lib(const char* lib_name,
                                                             std::int32_t flags = RTLD_LAZY |
                                                                                  RTLD_LOCAL)
{
  static_cast<void>(::dlerror());

  return {::dlopen(lib_name, flags), &::dlclose};
}

}  // namespace

SharedLibrary::SharedLibrary(std::string lib_path, std::int32_t flags, bool must_load)
  : handle_path_{std::move(lib_path)},
    handle_{load_lib(handle_path().data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
                     flags)}
{
  if (must_load && !is_loaded()) {
    // err might be NULL here, dlerror() is only required to return a valid error message when
    // dlsym() fails. In practice, however, dlerror() will return error messages for dlopen()
    // as well.
    const auto* err = ::dlerror();

    throw TracedException<std::runtime_error>{fmt::format(
      "Failed to load dynamic shared object {} ({})", handle_path(), err ? err : "unknown error")};
  }
}

SharedLibrary::SharedLibrary(std::string lib_path, bool must_load)
  : SharedLibrary{std::move(lib_path), RTLD_LAZY | RTLD_LOCAL, must_load}
{
}

SharedLibrary::SharedLibrary(std::nullptr_t) : handle_{load_lib(nullptr)} {}

void* SharedLibrary::load_symbol(ZStringView symbol_name) const
{
  if (!is_loaded()) {
    throw TracedException<std::runtime_error>{fmt::format(
      "Cannot load {}, failed to load shared library {} initially", symbol_name, handle_path())};
  }

  static_cast<void>(::dlerror());

  auto* const ret = ::dlsym(handle(),
                            symbol_name.data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
  );

  if (const char* error = ::dlerror(); error || !ret) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Failed to locate the symbol {} in the shared library: {}", symbol_name, error)};
  }

  if (ret && !resolved_path_) {
    if (::Dl_info info{}; ::dladdr(ret, &info)) {
      LEGATE_CHECK(info.dli_fname);
      handle_path_   = info.dli_fname;
      resolved_path_ = true;
    }
  }
  return ret;
}

// ------------------------------------------------------------------------------------------

/*static*/ bool SharedLibrary::exists(ZStringView lib_path)
{
  return load_lib(lib_path.data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
                  ) != nullptr;
}

}  // namespace legate::detail
