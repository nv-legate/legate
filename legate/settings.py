# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from .util.settings import (
    EnvOnlySetting,
    PrioritizedSetting,
    Settings,
    convert_bool,
    convert_int,
)

__all__ = ("settings",)


class LegateRuntimeSettings(Settings):
    consensus: EnvOnlySetting[bool] = EnvOnlySetting(
        "consensus",
        "LEGATE_CONSENSUS",
        default=False,
        test_default=False,
        convert=convert_bool,
        help="""
        Whether to perform the RegionField consensus match operation on
        single-node runs (for testing). This is normally only necessary on
        multi-node runs, where all processes must collectively agree that a
        RegionField has been garbage collected at the Python level before it
        can be reused.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    cycle_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "cycle_check",
        "LEGATE_CYCLE_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles involving RegionField objects on
        exit (developer option). When such cycles arise during execution they
        inhibit used RegionFields from being collected and reused for new
        Stores, thus increasing memory pressure. By default this check will
        miss any RegionField cycles that the garbage collector collected during
        execution. Run gc.disable() at the beginning of the program to avoid
        this.
        """,
    )

    future_leak_check: PrioritizedSetting[bool] = PrioritizedSetting(
        "future_leak_check",
        "LEGATE_FUTURE_LEAK_CHECK",
        default=False,
        convert=convert_bool,
        help="""
        Whether to check for reference cycles keeping Future/FutureMap objects
        alive after Legate runtime exit (developer option). Such leaks can
        result in Legion runtime shutdown hangs.
        """,
    )

    test: EnvOnlySetting[bool] = EnvOnlySetting(
        "test",
        "LEGATE_TEST",
        default=False,
        convert=convert_bool,
        help="""
        Enable test mode. This sets alternative defaults for various other
        settings.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    window_size: EnvOnlySetting[int] = EnvOnlySetting(
        "window_size",
        "LEGATE_WINDOW_SIZE",
        default=1,
        test_default=1,
        convert=convert_int,
        help="""
        How many Legate operations to accumulate before emitting to Legion.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_frac: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_frac",
        "LEGATE_FIELD_REUSE_FRAC",
        default=256,
        test_default=1,
        convert=convert_int,
        help="""
        Any allocation for more than 1/frac of available memory will count as
        multiple allocations, for purposes of triggering a consensus match.
        Only relevant for multi-node runs.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    field_reuse_freq: EnvOnlySetting[int] = EnvOnlySetting(
        "field_reuse_freq",
        "LEGATE_FIELD_REUSE_FREQ",
        default=32,
        test_default=8,
        convert=convert_int,
        help="""
        Every how many RegionField allocations to perform a consensus match
        operation. Only relevant for multi-node runs, where all processes must
        collectively agree that a RegionField has been garbage collected at the
        Python level before it can be reused.

        This is a read-only environment variable setting used by the runtime.
        """,
    )

    disable_mpi: EnvOnlySetting[bool] = EnvOnlySetting(
        "disable_mpi",
        "LEGATE_DISABLE_MPI",
        default=False,
        test_default=False,
        convert=convert_bool,
        help="""
        Disable the MPI-based communicator (used for collective operations in
        certain CPU tasks). Use this to work around MPI initialization
        failures.

        This is a read-only environment variable setting used by the runtime.
        """,
    )


settings = LegateRuntimeSettings()
