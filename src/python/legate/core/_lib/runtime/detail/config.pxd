# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint32_t, uint64_t
from libcpp cimport bool
from libcpp.optional cimport optional as std_optional
from libcpp.string cimport string as std_string
from ...utilities.unconstructable cimport Unconstructable

cdef extern from "legate/runtime/detail/config.h" namespace "legate" nogil:
    cdef cppclass _Config "legate::detail::Config":
        bool auto_config() except+
        bool show_progress_requested() except+
        bool use_empty_task() except+
        bool warmup_nccl() except+
        bool enable_inline_task_launch() except+
        bool single_controller_execution() except+
        uint32_t max_exception_size() except+
        uint64_t min_cpu_chunk() except+
        uint64_t min_gpu_chunk() except+
        uint64_t min_omp_chunk() except+
        uint32_t window_size() except+
        uint32_t field_reuse_frac() except+
        uint32_t field_reuse_freq() except+
        bool consensus() except+
        bool disable_mpi() except+
        bool io_use_vfd_gds() except+
        uint64_t num_omp_threads() except+
        bool profile() except+
        std_string profile_name() except+
        bool provenance() except+
        bool experimental_copy_path() except+

cdef extern from "legate/runtime/detail/config.h" \
  namespace "legate::detail" nogil:
    std_optional[T] _get_realm_config_property \
      "legate::detail::get_realm_config_property" [T] (
            const std_string &module_name,
            const std_string &property_name
    ) except+

cdef class Config(Unconstructable):
    cdef const _Config* _handle

    @staticmethod
    cdef Config from_handle(const _Config* handle)

cdef class RealmConfig:
    pass
