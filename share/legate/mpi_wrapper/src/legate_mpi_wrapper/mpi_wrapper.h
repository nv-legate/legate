/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */
// NOLINTBEGIN
// Must use C-isms here since mpi_wrapper.cc might be compiled by C compiler
#ifndef LEGATE_SHARE_LEGATE_MPI_WRAPPER_H
#define LEGATE_SHARE_LEGATE_MPI_WRAPPER_H

#include <legate_mpi_wrapper/mpi_wrapper_types.h>

#ifdef __cplusplus
#define LEGATE_EXTERN extern "C"
#else
// extern technically isn't required in C, but it cannot hurt
#define LEGATE_EXTERN extern
#endif

// NOLINTBEGIN
LEGATE_EXTERN Legate_MPI_Comm legate_mpi_comm_world(void);
LEGATE_EXTERN int legate_mpi_thread_multiple(void);
LEGATE_EXTERN int legate_mpi_tag_ub(void);
LEGATE_EXTERN int legate_mpi_congruent(void);
LEGATE_EXTERN int legate_mpi_success(void);

// ==========================================================================================

LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_int8_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_uint8_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_char(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_byte(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_int(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_int32_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_uint32_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_int64_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_uint64_t(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_float(void);
LEGATE_EXTERN Legate_MPI_Datatype legate_mpi_double(void);

// ==========================================================================================

LEGATE_EXTERN int legate_mpi_init(int* argc, char*** argv);
LEGATE_EXTERN int legate_mpi_init_thread(int* argc, char*** argv, int required, int* provided);
LEGATE_EXTERN int legate_mpi_finalize(void);
LEGATE_EXTERN int legate_mpi_abort(Legate_MPI_Comm comm, int error_code);
LEGATE_EXTERN int legate_mpi_initialized(int* init);
LEGATE_EXTERN int legate_mpi_finalized(int* finalized);
LEGATE_EXTERN int legate_mpi_comm_dup(Legate_MPI_Comm comm, Legate_MPI_Comm* dup);
LEGATE_EXTERN int legate_mpi_comm_rank(Legate_MPI_Comm comm, int* rank);
LEGATE_EXTERN int legate_mpi_comm_size(Legate_MPI_Comm comm, int* size);
LEGATE_EXTERN int legate_mpi_comm_compare(Legate_MPI_Comm comm1,
                                          Legate_MPI_Comm comm2,
                                          int* result);
LEGATE_EXTERN int legate_mpi_comm_get_attr(Legate_MPI_Comm comm,
                                           int comm_keyval,
                                           void* attribute_val,
                                           int* flag);
LEGATE_EXTERN int legate_mpi_comm_free(Legate_MPI_Comm* comm);
LEGATE_EXTERN int legate_mpi_type_get_extent(Legate_MPI_Datatype type,
                                             Legate_MPI_Aint* lb,
                                             Legate_MPI_Aint* extent);
LEGATE_EXTERN int legate_mpi_query_thread(int* provided);
LEGATE_EXTERN int legate_mpi_bcast(
  void* buffer, int count, Legate_MPI_Datatype datatype, int root, Legate_MPI_Comm comm);
LEGATE_EXTERN int legate_mpi_send(const void* buf,
                                  int count,
                                  Legate_MPI_Datatype datatype,
                                  int dest,
                                  int tag,
                                  Legate_MPI_Comm comm);

LEGATE_EXTERN int legate_mpi_recv(void* buf,
                                  int count,
                                  Legate_MPI_Datatype datatype,
                                  int source,
                                  int tag,
                                  Legate_MPI_Comm comm,
                                  Legate_MPI_Status* status);
LEGATE_EXTERN int legate_mpi_sendrecv(const void* sendbuf,
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
#endif  // LEGATE_SHARE_LEGATE_MPI_WRAPPER_H
// NOLINTEND
