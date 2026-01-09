/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/mpi_detection.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/shared_library.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <../../share/legate/mpi_wrapper/src/legate_mpi_wrapper/mpi_wrapper_types.h>
#include <dlfcn.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace legate::detail {

#define SHARED_LIBRARY(lib_name) LEGATE_SHARED_LIBRARY_PREFIX lib_name LEGATE_SHARED_LIBRARY_SUFFIX

namespace {

constexpr Legate_MPI_Kind LEGATE_MPI_KIND_UNKNOWN = -1;

[[nodiscard]] std::pair<Legate_MPI_Kind, std::optional<std::string>> detect_mpi_abi()
{
  // try to open libmpi.so and call MPI_Get_library_version to detect MPI version

  const auto mpi_handle = [&] {
    try {
      return SharedLibrary{SHARED_LIBRARY("mpi"),
#if LEGATE_DEFINED(LEGATE_LINUX)
                           // see https://github.com/jeffhammond/mukautuva/issues/32
                           RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND,
#else
                           RTLD_LAZY | RTLD_LOCAL,
#endif
                           /* must_load */ true};
    } catch (...) {
      throw TracedException<std::runtime_error>{
        fmt::format("Please make sure MPI is installed and {} is in your LD_LIBRARY_PATH.",
                    SHARED_LIBRARY("mpi"))};
      throw;
    }
  }();

  const auto mpi_get_library_version_fp =
    reinterpret_cast<int (*)(char*, int*)>(mpi_handle.load_symbol("MPI_Get_library_version"));

  // MPICH uses 8192: https://github.com/pmodels/mpich/blob/main/src/binding/abi/mpi_abi.h#L289
  // Open MPI uses 256: https://github.com/open-mpi/ompi/blob/main/ompi/include/mpi.h.in#L549
  // So we use 8192 as the default value to satisfy both MPICH and Open MPI
  constexpr int MPI_MAX_LIBRARY_VERSION_STRING     = 8192;
  char lib_version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
  int len{};

  const int rc = mpi_get_library_version_fp(lib_version, &len);

  if (rc != 0) {
    const auto mpi_error_string_fp =
      reinterpret_cast<int (*)(int, char*, int*)>(mpi_handle.load_symbol("MPI_Error_string"));
    constexpr int MPI_MAX_ERROR_STRING      = 512;
    char error_string[MPI_MAX_ERROR_STRING] = {0};

    if (mpi_error_string_fp != nullptr) {
      int tmp = 0;

      static_cast<void>(mpi_error_string_fp(rc, error_string, &tmp));
    }
    throw TracedException<std::runtime_error>{
      fmt::format("MPI_Get_library_version from {} failed: error: {}, error_string: {}",
                  mpi_handle.handle_path(),
                  rc,
                  error_string)};
  }

  const auto lib_version_view = std::string_view{lib_version, static_cast<std::size_t>(len)};

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

[[nodiscard]] Legate_MPI_Kind extract_mpi_type_from_wrapper(std::string wrapper_name)
{
  const auto handle = SharedLibrary{std::move(wrapper_name), /* must_load */ true};
  const auto legate_mpi_wrapper_kind_fp =
    reinterpret_cast<Legate_MPI_Kind (*)()>(handle.load_symbol("legate_mpi_wrapper_kind"));

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
  // if liblegate_mpi_wrapper.so exists, it is a local build, so we don't need to detect MPI
  // version
  if (SharedLibrary::exists(SHARED_LIBRARY("legate_mpi_wrapper"))) {
    log_legate().debug() << SHARED_LIBRARY("liblegate_mpi_wrapper")
                         << " exists, skipping MPI version detection";
    return;
  }

  auto mpi_abi_kind    = LEGATE_MPI_KIND_UNKNOWN;
  auto mpi_abi_version = std::optional<std::string>{std::nullopt};

  if (const auto wrapper = LEGATE_MPI_WRAPPER.get(); wrapper.has_value()) {
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
      throw TracedException<std::runtime_error>{
        fmt::format("Could not detect MPI version from version string: \"{}\". Please open an "
                    "issue at https://github.com/nv-legate/legate detailing your MPI installation",
                    mpi_abi_version.value_or(""))};
  }

  const auto wrapper    = LEGATE_MPI_WRAPPER.get();
  const auto plugin_opt = REALM_UCP_BOOTSTRAP_PLUGIN.get();

  LEGATE_CHECK(wrapper.has_value());
  LEGATE_CHECK(plugin_opt.has_value());
  log_legate().debug() << "The MPI used by the wrappers is: " << mpi_kind_to_string(mpi_abi_kind)
                       << ", LEGATE_MPI_WRAPPER is set to: "
                       << *wrapper  // NOLINT(bugprone-unchecked-optional-access)
                       << ", REALM_UCP_BOOTSTRAP_PLUGIN is set to: "
                       << *plugin_opt;  // NOLINT(bugprone-unchecked-optional-access)
}

#undef SHARED_LIBRARY

}  // namespace legate::detail
