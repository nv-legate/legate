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
#include "legate/comm/detail/mpi_interface.h"

#include "legate_defines.h"

#include "legate/utilities/detail/env.h"
#include "legate/utilities/detail/formatters.h"
#include "legate/utilities/detail/zstring_view.h"
#include "legate/utilities/macros.h"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/std.h>
#include <iterator>
#include <legate_mpi_wrapper/mpi_wrapper.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace legate::detail::comm::mpi::detail {

class MPIInterface::Impl {
 public:
  using MPI_Comm     = MPIInterface::MPI_Comm;
  using MPI_Datatype = MPIInterface::MPI_Datatype;
  using MPI_Aint     = MPIInterface::MPI_Aint;
  using MPI_Status   = MPIInterface::MPI_Status;

  Impl();

  // NOLINTBEGIN(readability-identifier-naming)
  MPI_Comm (*MPI_COMM_WORLD)() = nullptr;
  int (*MPI_THREAD_MULTIPLE)() = nullptr;
  int (*MPI_TAG_UB)()          = nullptr;
  int (*MPI_CONGRUENT)()       = nullptr;
  int (*MPI_SUCCESS)()         = nullptr;

  MPI_Datatype (*MPI_INT8_T)()   = nullptr;
  MPI_Datatype (*MPI_UINT8_T)()  = nullptr;
  MPI_Datatype (*MPI_CHAR)()     = nullptr;
  MPI_Datatype (*MPI_INT)()      = nullptr;
  MPI_Datatype (*MPI_BYTE)()     = nullptr;
  MPI_Datatype (*MPI_INT32_T)()  = nullptr;
  MPI_Datatype (*MPI_UINT32_T)() = nullptr;
  MPI_Datatype (*MPI_INT64_T)()  = nullptr;
  MPI_Datatype (*MPI_UINT64_T)() = nullptr;
  MPI_Datatype (*MPI_FLOAT)()    = nullptr;
  MPI_Datatype (*MPI_DOUBLE)()   = nullptr;
  // NOLINTEND(readability-identifier-naming)

  int (*mpi_init_thread)(int*, char***, int, int*) = nullptr;
  int (*mpi_finalize)()                            = nullptr;
  int (*mpi_abort)(MPI_Comm, int)                  = nullptr;

  int (*mpi_initialized)(int*) = nullptr;
  int (*mpi_finalized)(int*)   = nullptr;

  int (*mpi_comm_dup)(MPI_Comm, MPI_Comm*)             = nullptr;
  int (*mpi_comm_rank)(MPI_Comm, int*)                 = nullptr;
  int (*mpi_comm_size)(MPI_Comm, int*)                 = nullptr;
  int (*mpi_comm_compare)(MPI_Comm, MPI_Comm, int*)    = nullptr;
  int (*mpi_comm_get_attr)(MPI_Comm, int, void*, int*) = nullptr;
  int (*mpi_comm_free)(MPI_Comm*)                      = nullptr;

  int (*mpi_type_get_extent)(MPI_Datatype, MPI_Aint*, MPI_Aint*) = nullptr;

  int (*mpi_query_thread)(int*) = nullptr;

  int (*mpi_bcast)(void*, int, MPI_Datatype, int, MPI_Comm)                  = nullptr;
  int (*mpi_send)(const void*, int, MPI_Datatype, int, int, MPI_Comm)        = nullptr;
  int (*mpi_recv)(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status*) = nullptr;
  int (*mpi_sendrecv)(const void*,
                      int,
                      MPI_Datatype,
                      int,
                      int,
                      void*,
                      int,
                      MPI_Datatype,
                      int,
                      int,
                      MPI_Comm,
                      MPI_Status*)                                           = nullptr;

 private:
  [[nodiscard]] static std::filesystem::path get_wrapper_path_();
  [[nodiscard]] static std::optional<std::unique_ptr<void, int (*)(void*)>> try_load_handle_(
    const std::filesystem::path& path);
  [[nodiscard]] static std::unique_ptr<void, int (*)(void*)> load_handle_();
  void load_wrapper_();

  std::unique_ptr<void, int (*)(void*)> handle_;
};

// ==========================================================================================-

namespace {

[[nodiscard]] std::string read_wrapper_path_from_env()
{
  constexpr auto lib_name =
    ZStringView{LEGATE_SHARED_LIBRARY_PREFIX "legate_mpi_wrapper" LEGATE_SHARED_LIBRARY_SUFFIX};
  constexpr auto is_not_space = [](unsigned char ch) { return !std::isspace(ch); };

  auto ret = LEGATE_MPI_WRAPPER.get(/* default_value */ lib_name.as_string_view());

  // lstrip
  ret.erase(ret.begin(), std::find_if(ret.begin(), ret.end(), is_not_space));
  // rstrip
  ret.erase(std::find_if(ret.rbegin(), ret.rend(), is_not_space).base(), ret.end());
  if (ret.empty()) {
    // If the user passes LEGATE_MPI_WRAPPER='', default to the default wrapper name
    ret = lib_name.as_string_view();
  }
  return ret;
}

class InvalidWrapperLocation : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

class HandleLoadError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

}  // namespace

/*static*/ std::filesystem::path MPIInterface::Impl::get_wrapper_path_()
{
  auto orig_path = std::filesystem::path{read_wrapper_path_from_env()};
  // We do not canonicalize the path yet! The path (and, in fact, the default) is allowed to
  // just be a file name, in which case dlopen() will search the link and rpaths for the
  // library.
  if (std::distance(orig_path.begin(), orig_path.end()) == 1) {
    // path is e.g. liblegate_mpi_wrapper.so
    return orig_path;
  }

  auto resolved_path = std::filesystem::weakly_canonical(orig_path);

  if (!std::filesystem::exists(resolved_path)) {
    // Transform the path here so that the exception prints out a system-appropriate path in
    // case it does not exist. For example, on Windows, C:a/b/c/d *does* work, but a user
    // seeing such a path might think to themselves "well, obviously, you probably meant
    // C:a\b\c".
    throw InvalidWrapperLocation{fmt::format("invalid MPI wrapper location: '{}' (does not exist)",
                                             orig_path.make_preferred())};
  }
  resolved_path.make_preferred();
  return resolved_path;
}

/*static*/ std::optional<std::unique_ptr<void, int (*)(void*)>>
MPIInterface::Impl::try_load_handle_(const std::filesystem::path& path)
{
  static_cast<void>(::dlerror());
  if (auto* const handle = ::dlopen(path.c_str(), RTLD_NOW)) {
    return {{handle, &(::dlclose)}};
  }
  return std::nullopt;
}

/*static*/ std::unique_ptr<void, int (*)(void*)> MPIInterface::Impl::load_handle_()
{
  const auto wrapper_lib = get_wrapper_path_();

  if (auto handle = try_load_handle_(wrapper_lib); handle.has_value()) {
    return *std::move(handle);
  }

  throw HandleLoadError{
    fmt::format("failed to load MPI wrapper '{}': {}", wrapper_lib, ::dlerror())};
}

void MPIInterface::Impl::load_wrapper_()
{
#define LEGATE_LOAD_FN(dest, src)                                                             \
  do {                                                                                        \
    using dest_type = std::decay_t<decltype(this->dest)>;                                     \
    static_assert(std::is_same_v<dest_type, std::decay_t<decltype(src)>>);                    \
                                                                                              \
    static_cast<void>(::dlerror());                                                           \
                                                                                              \
    if (const auto ret = ::dlsym(this->handle_.get(), LEGATE_STRINGIZE(src))) {               \
      this->dest = reinterpret_cast<dest_type>(ret); /* NOLINT(bugprone-macro-parentheses) */ \
    } else {                                                                                  \
      throw std::runtime_error{                                                               \
        fmt::format("dlsym(" LEGATE_STRINGIZE(src) ") failed: {}", ::dlerror())};             \
    }                                                                                         \
  } while (0)

  LEGATE_LOAD_FN(MPI_COMM_WORLD, legate_mpi_comm_world);
  LEGATE_LOAD_FN(MPI_THREAD_MULTIPLE, legate_mpi_thread_multiple);
  LEGATE_LOAD_FN(MPI_TAG_UB, legate_mpi_tag_ub);
  LEGATE_LOAD_FN(MPI_CONGRUENT, legate_mpi_congruent);
  LEGATE_LOAD_FN(MPI_SUCCESS, legate_mpi_success);

  LEGATE_LOAD_FN(MPI_INT8_T, legate_mpi_int8_t);
  LEGATE_LOAD_FN(MPI_UINT8_T, legate_mpi_uint8_t);
  LEGATE_LOAD_FN(MPI_CHAR, legate_mpi_char);
  LEGATE_LOAD_FN(MPI_BYTE, legate_mpi_byte);
  LEGATE_LOAD_FN(MPI_INT, legate_mpi_int);
  LEGATE_LOAD_FN(MPI_INT32_T, legate_mpi_int32_t);
  LEGATE_LOAD_FN(MPI_UINT32_T, legate_mpi_uint32_t);
  LEGATE_LOAD_FN(MPI_INT64_T, legate_mpi_int64_t);
  LEGATE_LOAD_FN(MPI_UINT64_T, legate_mpi_uint64_t);
  LEGATE_LOAD_FN(MPI_FLOAT, legate_mpi_float);
  LEGATE_LOAD_FN(MPI_DOUBLE, legate_mpi_double);

  LEGATE_LOAD_FN(mpi_init_thread, legate_mpi_init_thread);
  LEGATE_LOAD_FN(mpi_finalize, legate_mpi_finalize);
  LEGATE_LOAD_FN(mpi_abort, legate_mpi_abort);
  LEGATE_LOAD_FN(mpi_initialized, legate_mpi_initialized);
  LEGATE_LOAD_FN(mpi_finalized, legate_mpi_finalized);
  LEGATE_LOAD_FN(mpi_comm_dup, legate_mpi_comm_dup);
  LEGATE_LOAD_FN(mpi_comm_rank, legate_mpi_comm_rank);
  LEGATE_LOAD_FN(mpi_comm_size, legate_mpi_comm_size);
  LEGATE_LOAD_FN(mpi_comm_compare, legate_mpi_comm_compare);
  LEGATE_LOAD_FN(mpi_comm_get_attr, legate_mpi_comm_get_attr);
  LEGATE_LOAD_FN(mpi_comm_free, legate_mpi_comm_free);
  LEGATE_LOAD_FN(mpi_type_get_extent, legate_mpi_type_get_extent);
  LEGATE_LOAD_FN(mpi_query_thread, legate_mpi_query_thread);
  LEGATE_LOAD_FN(mpi_bcast, legate_mpi_bcast);
  LEGATE_LOAD_FN(mpi_send, legate_mpi_send);
  LEGATE_LOAD_FN(mpi_recv, legate_mpi_recv);
  LEGATE_LOAD_FN(mpi_sendrecv, legate_mpi_sendrecv);

#undef LEGATE_LOAD_FN
}

// ==========================================================================================

MPIInterface::Impl::Impl() : handle_{load_handle_()} { load_wrapper_(); }

// ==========================================================================================

/*static*/ const MPIInterface::Impl& MPIInterface::get_interface_()
{
  static const Impl iface{};

  return iface;
}

// ==========================================================================================

/*static*/ MPIInterface::MPI_Comm MPIInterface::MPI_COMM_WORLD()
{
  return get_interface_().MPI_COMM_WORLD();
}

/*static*/ int MPIInterface::MPI_THREAD_MULTIPLE()
{
  return get_interface_().MPI_THREAD_MULTIPLE();
}

/*static*/ int MPIInterface::MPI_TAG_UB() { return get_interface_().MPI_TAG_UB(); }

/*static*/ int MPIInterface::MPI_CONGRUENT() { return get_interface_().MPI_CONGRUENT(); }

/*static*/ int MPIInterface::MPI_SUCCESS() { return get_interface_().MPI_SUCCESS(); }

// ==========================================================================================

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT8_T()
{
  return get_interface_().MPI_INT8_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT8_T()
{
  return get_interface_().MPI_UINT8_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_CHAR()
{
  return get_interface_().MPI_CHAR();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_BYTE()
{
  return get_interface_().MPI_BYTE();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT() { return get_interface_().MPI_INT(); }

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT32_T()
{
  return get_interface_().MPI_INT32_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT32_T()
{
  return get_interface_().MPI_UINT32_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_INT64_T()
{
  return get_interface_().MPI_INT64_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_UINT64_T()
{
  return get_interface_().MPI_UINT64_T();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_FLOAT()
{
  return get_interface_().MPI_FLOAT();
}

/*static*/ MPIInterface::MPI_Datatype MPIInterface::MPI_DOUBLE()
{
  return get_interface_().MPI_DOUBLE();
}

// ==========================================================================================

/*static*/ int MPIInterface::mpi_init_thread(int* argc, char*** argv, int required, int* provided)
{
  return get_interface_().mpi_init_thread(argc, argv, required, provided);
}

/*static*/ int MPIInterface::mpi_finalize() { return get_interface_().mpi_finalize(); }

/*static*/ int MPIInterface::mpi_abort(MPIInterface::MPI_Comm comm, int error_code)
{
  return get_interface_().mpi_abort(comm, error_code);
}

/*static*/ int MPIInterface::mpi_initialized(int* init)
{
  return get_interface_().mpi_initialized(init);
}

/*static*/ int MPIInterface::mpi_finalized(int* finalized)
{
  return get_interface_().mpi_finalized(finalized);
}

/*static*/ int MPIInterface::mpi_comm_dup(MPIInterface::MPI_Comm comm, MPIInterface::MPI_Comm* dup)
{
  return get_interface_().mpi_comm_dup(comm, dup);
}

/*static*/ int MPIInterface::mpi_comm_rank(MPIInterface::MPI_Comm comm, int* rank)
{
  return get_interface_().mpi_comm_rank(comm, rank);
}

/*static*/ int MPIInterface::mpi_comm_size(MPIInterface::MPI_Comm comm, int* size)
{
  return get_interface_().mpi_comm_size(comm, size);
}

/*static*/ int MPIInterface::mpi_comm_compare(MPIInterface::MPI_Comm comm1,
                                              MPIInterface::MPI_Comm comm2,
                                              int* result)
{
  return get_interface_().mpi_comm_compare(comm1, comm2, result);
}

/*static*/ int MPIInterface::mpi_comm_get_attr(MPIInterface::MPI_Comm comm,
                                               int comm_keyval,
                                               void* attribute_val,
                                               int* flag)
{
  return get_interface_().mpi_comm_get_attr(comm, comm_keyval, attribute_val, flag);
}

/*static*/ int MPIInterface::mpi_comm_free(MPIInterface::MPI_Comm* comm)
{
  return get_interface_().mpi_comm_free(comm);
}

/*static*/ int MPIInterface::mpi_type_get_extent(MPIInterface::MPI_Datatype type,
                                                 MPIInterface::MPI_Aint* lb,
                                                 MPIInterface::MPI_Aint* extent)
{
  return get_interface_().mpi_type_get_extent(type, lb, extent);
}

/*static*/ int MPIInterface::mpi_query_thread(int* provided)
{
  return get_interface_().mpi_query_thread(provided);
}

/*static*/ int MPIInterface::mpi_bcast(void* buffer,
                                       int count,
                                       MPIInterface::MPI_Datatype datatype,
                                       int root,
                                       MPIInterface::MPI_Comm comm)
{
  return get_interface_().mpi_bcast(buffer, count, datatype, root, comm);
}

/*static*/ int MPIInterface::mpi_send(const void* buf,
                                      int count,
                                      MPIInterface::MPI_Datatype datatype,
                                      int dest,
                                      int tag,
                                      MPIInterface::MPI_Comm comm)
{
  return get_interface_().mpi_send(buf, count, datatype, dest, tag, comm);
}

/*static*/ int MPIInterface::mpi_recv(void* buf,
                                      int count,
                                      MPIInterface::MPI_Datatype datatype,
                                      int source,
                                      int tag,
                                      MPIInterface::MPI_Comm comm,
                                      MPIInterface::MPI_Status* status)
{
  return get_interface_().mpi_recv(buf, count, datatype, source, tag, comm, status);
}

/*static*/ int MPIInterface::mpi_sendrecv(const void* sendbuf,
                                          int sendcount,
                                          MPIInterface::MPI_Datatype sendtype,
                                          int dest,
                                          int sendtag,
                                          void* recvbuf,
                                          int recvcount,
                                          MPIInterface::MPI_Datatype recvtype,
                                          int source,
                                          int recvtag,
                                          MPIInterface::MPI_Comm comm,
                                          MPIInterface::MPI_Status* status)
{
  return get_interface_().mpi_sendrecv(sendbuf,
                                       sendcount,
                                       sendtype,
                                       dest,
                                       sendtag,
                                       recvbuf,
                                       recvcount,
                                       recvtype,
                                       source,
                                       recvtag,
                                       comm,
                                       status);
}

}  // namespace legate::detail::comm::mpi::detail
