/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

// Must use C-isms here since mpi_wrapper.cc might be compiled by C compiler
// NOLINTBEGIN
#ifndef LEGATE_SHARE_LEGATE_MPI_WRAPPER_TYPES_H
#define LEGATE_SHARE_LEGATE_MPI_WRAPPER_TYPES_H

#include <stddef.h>

typedef ptrdiff_t Legate_MPI_Comm;
typedef ptrdiff_t Legate_MPI_Datatype;
typedef ptrdiff_t Legate_MPI_Aint;

// The size in bytes of the thunk where we will stash the original MPI_Status. While the
// standard mandates the public members of MPI_Status, it says nothing about the order, or
// layout of the rest of the struct. And, of course, both MPICH and OpenMPI vary greatly:
//
// MPICH:
// https://github.com/pmodels/mpich/blob/29c640a0d6533424a6afbf644d14a9a5f7a1c870/src/include/mpi.h.in#L378
// OpenMPI:
// https://github.com/open-mpi/ompi/blob/1438a792caca3e2c862982d80d6d0fb403658e15/ompi/include/mpi.h.in#L468
//
// So the strategy is to simply embed the true MPI_Status structure inside ours. 64 bytes
// should be large enough for any reasonable MPI implementation of it.
#define LEGATE_MPI_STATUS_THUNK_SIZE 64

typedef struct Legate_MPI_Status {
  int MPI_SOURCE;
  int MPI_TAG;
  int MPI_ERROR;
  char original_private_[LEGATE_MPI_STATUS_THUNK_SIZE];
} Legate_MPI_Status;

#endif  // LEGATE_SHARE_LEGATE_MPI_WRAPPER_TYPES_H
// NOLINTEND
