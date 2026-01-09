/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
// NOLINTBEGIN
// Must use C-isms here since mpi_wrapper.cc might be compiled by C compiler
#ifndef LEGATE_SHARE_LEGATE_MPI_WRAPPER_H  // legate-lint: no-legate-defined
#define LEGATE_SHARE_LEGATE_MPI_WRAPPER_H

#include <legate_mpi_wrapper/mpi_wrapper_types.h>

// Set by Legate CMake if we are building with networks but without MPI. In this case, legate
// just needs this header to be include-able, but doesn't care about symbol visibility because
// it will never attempt to load the symbols.
#ifdef LEGATE_MPI_WRAPPER_HAVE_NO_EXPORT_HEADER  // legate-lint: no-legate-defined
#define LEGATE_MPI_WRAPPER_EXPORT
#else
#include <legate_mpi_wrapper/legate_mpi_wrapper_exports.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Kind legate_mpi_wrapper_kind(void);

// ==========================================================================================

LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Comm legate_mpi_comm_world(void);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_thread_multiple(void);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_tag_ub(void);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_congruent(void);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_success(void);

// ==========================================================================================

LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_int8_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_uint8_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_char(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_byte(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_int(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_int32_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_uint32_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_int64_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_uint64_t(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_float(void);
LEGATE_MPI_WRAPPER_EXPORT Legate_MPI_Datatype legate_mpi_double(void);

// ==========================================================================================

LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_init(int* argc, char*** argv);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_init_thread(int* argc,
                                                     char*** argv,
                                                     int required,
                                                     int* provided);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_finalize(void);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_abort(Legate_MPI_Comm comm, int error_code);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_initialized(int* init);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_finalized(int* finalized);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_dup(Legate_MPI_Comm comm, Legate_MPI_Comm* dup);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_rank(Legate_MPI_Comm comm, int* rank);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_size(Legate_MPI_Comm comm, int* size);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_compare(Legate_MPI_Comm comm1,
                                                      Legate_MPI_Comm comm2,
                                                      int* result);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_get_attr(Legate_MPI_Comm comm,
                                                       int comm_keyval,
                                                       void* attribute_val,
                                                       int* flag);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_comm_free(Legate_MPI_Comm* comm);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_type_get_extent(Legate_MPI_Datatype type,
                                                         Legate_MPI_Aint* lb,
                                                         Legate_MPI_Aint* extent);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_query_thread(int* provided);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_bcast(
  void* buffer, int count, Legate_MPI_Datatype datatype, int root, Legate_MPI_Comm comm);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_send(const void* buf,
                                              int count,
                                              Legate_MPI_Datatype datatype,
                                              int dest,
                                              int tag,
                                              Legate_MPI_Comm comm);

LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_recv(void* buf,
                                              int count,
                                              Legate_MPI_Datatype datatype,
                                              int source,
                                              int tag,
                                              Legate_MPI_Comm comm,
                                              Legate_MPI_Status* status);
LEGATE_MPI_WRAPPER_EXPORT int legate_mpi_sendrecv(const void* sendbuf,
                                                  int sendcount,
                                                  Legate_MPI_Datatype sendtype,
                                                  int dest,
                                                  int sendtag,
                                                  void* recvbuf,
                                                  int recvcount,
                                                  Legate_MPI_Datatype recvtype,
                                                  int source,
                                                  int recvtag,
                                                  Legate_MPI_Comm comm,
                                                  Legate_MPI_Status* status);

#ifdef __cplusplus
}
#endif

#endif  // LEGATE_SHARE_LEGATE_MPI_WRAPPER_H
// NOLINTEND
