# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport int32_t, int64_t

from ...utilities.unconstructable cimport Unconstructable

cdef class Config(Unconstructable):
    """
    Configuration access interface.

    .. warning::
        This class is considered an implementation detail and its members
        have no guarantee of stability across versions.
    """

    @staticmethod
    cdef Config from_handle(const _Config* handle):
        cdef Config result = Config.__new__(Config)
        result._handle = handle
        return result

    @property
    def auto_config(self) -> bool:
        r"""
        Whether legate attempts to automatically detect a suitable configuration.

        :rtype: bool
        """
        return self._handle.auto_config()

    @property
    def show_progress_requested(self) -> bool:
        r"""
        Whether legate prints a progress summary before each task is executed.

        :rtype: bool
        """
        return self._handle.show_progress_requested()

    @property
    def use_empty_task(self) -> bool:
        r"""
        Whether legate executes a dummy tasks in place of each task execution
        (primarily a developer feature for use in debugging, not recommended
        for exteranl use).

        :rtype: bool
        """
        return self._handle.use_empty_task()

    @property
    def warmup_nccl(self) -> bool:
        r"""
        Whether legate performs a warmup computation for NCCL on startup.

        :rtype: bool
        """
        return self._handle.warmup_nccl()

    @property
    def enable_inline_task_launch(self) -> bool:
        r"""
        Whether single controller execution is enabled, in which the top-level
        task only runs on rank 0.

        :rtype: bool
        """
        return self._handle.enable_inline_task_launch()

    @property
    def single_controller_execution(self) -> bool:
        r"""
        Whether inline task launching is used (experimental, set
        LEGATE_CONFIG="--help" for information on this option).

        :rtype: bool
        """
        return self._handle.single_controller_execution()

    @property
    def max_exception_size(self) -> int:
        r"""
        Maximum size (in bytes) to allocate for exception messages.

        :rtype: int
        """
        return int(self._handle.max_exception_size())

    @property
    def min_cpu_chunk(self) -> int:
        r"""
        Minimum CPU chunk size (in bytes).

        If using CPUs, any task operating on arrays smaller than this will not
        be parallelized across more than one core.

        :rtype: int
        """
        return int(self._handle.min_cpu_chunk())

    @property
    def min_gpu_chunk(self) -> int:
        r"""
        Minimum GPU chunk size (in bytes).

        If using GPUs, any task operating on arrays smaller than this will not
        be parallelized across more than one GPU.

        :rtype: int
        """
        return int(self._handle.min_gpu_chunk())

    @property
    def min_omp_chunk(self) -> int:
        r"""
        Minimum OpenMP chunk size (in bytes).

        If using OpenMP, any task operating on arrays smaller than this will not
        be parallelized across more than one OpenMP group.

        :rtype: int
        """
        return int(self._handle.min_omp_chunk())

    @property
    def window_size(self) -> int:
        r"""
        Maximum size of the submitted operation queue before forced flush.

        :rtype: int
        """
        return int(self._handle.window_size())

    @property
    def field_reuse_frac(self) -> int:
        r"""
        The amount (in bytes) of the "primary" memory type allocated by legate
        before consensus match is triggered.

        (Set LEGATE_CONFIG="--help" for for information on this option.)

        :rtype: int
        """
        return int(self._handle.field_reuse_frac())

    @property
    def field_reuse_freq(self) -> int:
        r"""
        The size (in number of stores) of the discarded store/array field cache
        retained by legate.

        :rtype: int
        """
        return int(self._handle.field_reuse_freq())

    @property
    def consensus(self) -> bool:
        r"""
        Whether legate performs the RegionField consensus match operation on
        single-node runs (for testing, primarily a developer feature for
        debugging).

        :rtype: int
        """
        return self._handle.consensus()

    @property
    def disable_mpi(self) -> bool:
        r"""
        Whether legate has disabled MPI support.

        This is useful if Legate was configured with MPI support (which usually
        causes Legate to use it), but MPI is not functional on the current system.
        When this flag is passed, no task should be launched that requests the MPI
        communicator, or the program will fail.

        :rtype: bool
        """
        return self._handle.disable_mpi()

    @property
    def io_use_vfd_gds(self) -> bool:
        r"""
        Whether legate has enabled HDF5 Virtual File Driver (VDS)
        GPUDirectStorage (GDS) for I/O.

        :rtype: bool
        """
        return self._handle.io_use_vfd_gds()

    @property
    def num_omp_threads(self) -> int:
        r"""
        Number of OpenMP groups used per rank by legate.

        (Set LEGATE_CONFIG="--help" for for information on this option.)

        :rtype: int
        """
        return int(self._handle.num_omp_threads())

    @property
    def profile(self) -> bool:
        r"""
        Whether profiling is enabled.

        :rtype: bool
        """
        return self._handle.profile()

    @property
    def profile_name(self) -> str:
        r"""
        Base filename for profiling logs.

        Legate create one file per rank (<profile-name>_<rank>.prof) relative
        to the log directory.

        :rtype: str
        """
        return self._handle.profile_name().decode("utf-8")

    @property
    def provenance(self) -> bool:
        r"""
        Whether provenance tracking is enabled.

        :rtype: bool
        """
        return self._handle.provenance()

    @property
    def experimental_copy_path(self) -> bool:
        r"""
        Whether conditional copy optimizations based on workload
        characteristics is enabled (experimental).

        :rtype: bool
        """
        return self._handle.experimental_copy_path()


ctypedef fused ConfigType:
    int32_t
    int64_t

cdef int32_t INT32_T = -1
cdef int64_t INT64_T = -1

cdef object get_realm_config_property(
    ConfigType T,
    str module_name,
    str property_name,
):
    cdef std_optional[ConfigType] v
    # in most cases cython can handle transforming str to a std::string, but
    # without this explicit transformation to bytes a runtime error occurs:
    #
    #       File "<stringsource>", line 15, in
    #         string.from_py
    #           .__pyx_convert_string_from_py_6libcpp_6string_std__in_string
    #     TypeError: expected bytes, str found
    mod = bytes(module_name, "utf-8")
    prop = bytes(property_name, "utf-8")
    if ConfigType is int32_t:
        v = _get_realm_config_property[int32_t](mod, prop)
    else:
        assert ConfigType is int64_t
        v = _get_realm_config_property[int64_t](mod, prop)
    if v.has_value():
        return int(v.value())
    return None


cdef class RealmConfig:
    """
    Realm configuration access interface.

    .. warning::
        This class is considered an implementation detail and its members
        have no guarantee of stability across versions.
    """

    @property
    def cpus(self) -> int | None:
        r"""
        Number of standalone CPU cores reserved per rank.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "core", "cpu")

    @property
    def util(self) -> int | None:
        r"""
        Number of threads per rank used for runtime meta-work.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "core", "util")

    @property
    def sysmem(self) -> int | None:
        r"""
        Size in bytes of DRAM memory reserved per rank.

        :rtype: int | None
        """
        return get_realm_config_property(INT64_T, "core", "sysmem")

    @property
    def regmem(self) -> int | None:
        r"""
        Size in bytes of reserved NIC-registered DRAM memory.

        :rtype: int | None
        """
        return get_realm_config_property(INT64_T, "core", "regmem")

    @property
    def gpus(self) -> int | None:
        r"""
        Number of GPUs reserved per rank.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "cuda", "cpu")

    @property
    def fbmem(self) -> int | None:
        r"""
        Size in bytes of GPU (or "framebuffer") memory reserved per GPU.

        :rtype: int | None
        """
        return get_realm_config_property(INT64_T, "cuda", "fbmem")

    @property
    def zcmem(self) -> int | None:
        r"""
        Size in bytes of GPU-registered (or "zero-copy") DRAM memory reserved
        per GPU.

        :rtype: int | None
        """
        return get_realm_config_property(INT64_T, "cuda", "zcmem")

    @property
    def omps(self) -> int | None:
        r"""
        Number of OpenMP groups per rank.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "openmp", "ocpu")

    @property
    def ompthreads(self) -> int | None:
        r"""
        Number of threads (reserved GPU cores) per OpenMP group.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "openmp", "othr")

    @property
    def numamem(self) -> int | None:
        r"""
        Size in bytes of NUMA-specific DRAM memory to reserve per NUMA domain.

        :rtype: int | None
        """
        return get_realm_config_property(INT32_T, "numa", "numamem")
