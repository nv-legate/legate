/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/mpi_detection.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <../../share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper_types.h>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace legate::detail {

#define SHARED_LIBRARY(lib_name) LEGATE_SHARED_LIBRARY_PREFIX lib_name LEGATE_SHARED_LIBRARY_SUFFIX

#define LEGATE_MPI_KIND_UNKNOWN ((Legate_MPI_Kind) - 1)

namespace {

[[nodiscard]] bool library_exists(ZStringView lib_name)
{
  const char* lib_name_str = lib_name.as_string_view().data();

  if (lib_name_str == nullptr) {
    return false;
  }
  const std::unique_ptr<void, int (*)(void*)> handle{::dlopen(lib_name_str, RTLD_LAZY | RTLD_LOCAL),
                                                     &::dlclose};

  return handle != nullptr;
}

[[nodiscard]] std::string get_library_path(const void* symbol)
{
  if (::Dl_info info{}; ::dladdr(symbol, &info) && info.dli_fname) {
    return info.dli_fname;
  }
  return "";
}

[[nodiscard]] std::pair<Legate_MPI_Kind, std::optional<std::string>> detect_mpi_abi()
{
  // try to open libmpi.so and call MPI_Get_library_version to detect MPI version
  // clear any existing error
  static_cast<void>(::dlerror());
  const std::unique_ptr<void, int (*)(void*)> mpi_handle{
#if LEGATE_DEFINED(LEGATE_LINUX)
    ::dlopen(SHARED_LIBRARY("mpi"), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND), &::dlclose
#else
    ::dlopen(SHARED_LIBRARY("mpi"), RTLD_LAZY | RTLD_LOCAL), &::dlclose
#endif
  };

  if (mpi_handle == nullptr) {
    const char* error_msg = ::dlerror();
    if (!error_msg) {
      error_msg = "Unknown error occurred while loading libmpi.so";
    }
    throw TracedException<std::runtime_error>{
      fmt::format("dlopen(\"{}\") failed: {}, please make sure MPI is installed and {} "
                  "is in your LD_LIBRARY_PATH.",
                  SHARED_LIBRARY("mpi"),
                  error_msg,
                  SHARED_LIBRARY("mpi"))};
  }

  static_cast<void>(::dlerror());
  const auto mpi_get_library_version_fp =
    reinterpret_cast<int (*)(char*, int*)>(::dlsym(mpi_handle.get(), "MPI_Get_library_version"));

  if (mpi_get_library_version_fp == nullptr) {
    const char* error_msg = ::dlerror();
    if (!error_msg) {
      error_msg = "unknown error";
    }
    throw TracedException<std::runtime_error>{
      fmt::format("dlsym(\"MPI_Get_library_version\") failed: {}", error_msg)};
  }

  // MPICH uses 8192: https://github.com/pmodels/mpich/blob/main/src/binding/abi/mpi_abi.h#L289
  // Open MPI uses 256: https://github.com/open-mpi/ompi/blob/main/ompi/include/mpi.h.in#L549
  // So we use 8192 as the default value to satisfy both MPICH and Open MPI
  constexpr int MPI_MAX_LIBRARY_VERSION_STRING     = 8192;
  char lib_version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
  int lib_version_length{};
  const int rc = mpi_get_library_version_fp(lib_version, &lib_version_length);

  if (rc != 0) {
    const auto mpi_error_string_fp =
      reinterpret_cast<int (*)(int, char*, int*)>(::dlsym(mpi_handle.get(), "MPI_Error_string"));
    constexpr int MPI_MAX_ERROR_STRING      = 512;
    char error_string[MPI_MAX_ERROR_STRING] = {0};

    if (mpi_error_string_fp != nullptr) {
      int length_of_error_string = 0;
      mpi_error_string_fp(rc, error_string, &length_of_error_string);
    }
    throw TracedException<std::runtime_error>{
      fmt::format("MPI_Get_library_version from {} failed: error: {}, error_string: {}",
                  get_library_path(reinterpret_cast<void*>(mpi_get_library_version_fp)),
                  rc,
                  error_string)};
  }

  const std::string_view lib_version_view{lib_version,
                                          static_cast<std::size_t>(lib_version_length)};

  if (lib_version_view.find("Open MPI") != std::string_view::npos) {
    return {LEGATE_MPI_KIND_OPEN_MPI, std::nullopt};
  }

  if (lib_version_view.find("MPICH") != std::string_view::npos ||
      lib_version_view.find("Intel(R) MPI Library") != std::string_view::npos ||
      lib_version_view.find("MVAPICH") != std::string_view::npos) {
    return {LEGATE_MPI_KIND_MPICH, std::nullopt};
  }

  return {LEGATE_MPI_KIND_UNKNOWN, lib_version};
}

[[nodiscard]] Legate_MPI_Kind extract_mpi_type_from_wrapper(ZStringView wrapper_name)
{
  const std::string wrapper_name_str(wrapper_name.as_string_view());
  const std::unique_ptr<void, int (*)(void*)> handle{
    ::dlopen(wrapper_name_str.c_str(), RTLD_LAZY | RTLD_LOCAL), &::dlclose};

  if (handle == nullptr) {
    return LEGATE_MPI_KIND_UNKNOWN;
  }

  const auto legate_mpi_wrapper_kind_fp =
    reinterpret_cast<std::int32_t (*)()>(::dlsym(handle.get(), "legate_mpi_wrapper_kind"));

  if (legate_mpi_wrapper_kind_fp == nullptr) {
    return LEGATE_MPI_KIND_UNKNOWN;
  }

  return legate_mpi_wrapper_kind_fp();
}

[[nodiscard]] std::string_view mpi_kind_to_string(Legate_MPI_Kind kind)
{
  switch (kind) {
    case LEGATE_MPI_KIND_MPICH: return "MPICH";
    case LEGATE_MPI_KIND_OPEN_MPI: return "Open MPI";
    default: break;  // legate-lint: no-switch-default
  }
  LEGATE_ABORT("Unknown MPI kind: ", kind);
}

}  // namespace

void set_mpi_wrapper_libraries()
{
  // if liblegate_mpi_wrapper.so exists, it is a local build, so we
  // don't need to detect MPI version
  if (library_exists(SHARED_LIBRARY("legate_mpi_wrapper"))) {
    log_legate().debug() << "liblegate_mpi_wrapper.so exists, skipping MPI version detection";
    return;
  }

  auto wrapper         = LEGATE_MPI_WRAPPER.get();
  auto mpi_abi_kind    = LEGATE_MPI_KIND_UNKNOWN;
  auto mpi_abi_version = std::optional<std::string>{std::nullopt};

  if (wrapper.has_value()) {
    // if LEGATE_MPI_WRAPPER is set, we get the MPI type from the wrapper
    mpi_abi_kind = extract_mpi_type_from_wrapper(*wrapper);
  } else {
    // detect MPI ABI type
    std::tie(mpi_abi_kind, mpi_abi_version) = detect_mpi_abi();
  }

  constexpr EnvironmentVariable<std::string> REALM_UCP_BOOTSTRAP_PLUGIN{
    "REALM_UCP_BOOTSTRAP_PLUGIN"};

  switch (mpi_abi_kind) {
    case LEGATE_MPI_KIND_OPEN_MPI:
      REALM_UCP_BOOTSTRAP_PLUGIN.set(SHARED_LIBRARY("realm_ucp_bootstrap_mpi_ompi"),
                                     /* overwrite */ false);
      LEGATE_MPI_WRAPPER.set(SHARED_LIBRARY("legate_mpi_wrapper_ompi"), /* overwrite */ false);
      break;
    case LEGATE_MPI_KIND_MPICH:
      REALM_UCP_BOOTSTRAP_PLUGIN.set(SHARED_LIBRARY("realm_ucp_bootstrap_mpi_mpich"),
                                     /* overwrite */ false);
      LEGATE_MPI_WRAPPER.set(SHARED_LIBRARY("legate_mpi_wrapper_mpich"), /* overwrite */ false);
      break;
    default:  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{fmt::format(
        "Unknown MPI version: {}, the LEGATE_MPI_WRAPPER should be either _ompi or _mpich.",
        mpi_abi_version.value_or(""))};
  }
  const auto wrapper_opt = LEGATE_MPI_WRAPPER.get();
  LEGATE_CHECK(wrapper_opt.has_value());
  const auto plugin_opt = REALM_UCP_BOOTSTRAP_PLUGIN.get();
  LEGATE_CHECK(plugin_opt.has_value());
  log_legate().debug() << "The MPI used by the wrappers is: " << mpi_kind_to_string(mpi_abi_kind)
                       << ", LEGATE_MPI_WRAPPER is set to: " << *wrapper_opt
                       << ", REALM_UCP_BOOTSTRAP_PLUGIN is set to: " << *plugin_opt;
}

#undef SHARED_LIBRARY

}  // namespace legate::detail
