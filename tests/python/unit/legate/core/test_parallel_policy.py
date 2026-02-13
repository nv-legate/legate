# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import pytest

from legate.core import ParallelPolicy, Scope, StreamingMode, TaskTarget

OVERDECOMPOSE_FACTOR = 42

# arbitrary values that we are less likely to set as default
CPU_THRESHOLD = 9
OMP_THRESHOLD = 17
GPU_THRESHOLD = 53


class TestParallelPolicy:
    def test_default_policy(self) -> None:
        a = ParallelPolicy()
        b = ParallelPolicy()
        assert a == b

    def test_default_scope_policy(self) -> None:
        a = Scope.parallel_policy()
        b = Scope.parallel_policy()
        assert a == b

    def test_default_policy_setting(self) -> None:
        a = ParallelPolicy()
        assert a.overdecompose_factor == 1
        assert not a.streaming

    def test_overdecompose_factor(self) -> None:
        a = ParallelPolicy(overdecompose_factor=OVERDECOMPOSE_FACTOR)
        assert a.overdecompose_factor == OVERDECOMPOSE_FACTOR

    def test_streaming(self) -> None:
        a = ParallelPolicy(streaming_mode=StreamingMode.RELAXED)
        assert a.streaming

    def test_streaming_and_overdecomposition_factor(self) -> None:
        a = ParallelPolicy(
            streaming_mode=StreamingMode.STRICT,
            overdecompose_factor=OVERDECOMPOSE_FACTOR,
        )
        assert a.streaming
        assert a.overdecompose_factor == OVERDECOMPOSE_FACTOR

    def test_comparison(self) -> None:
        a = ParallelPolicy()
        b = ParallelPolicy(streaming_mode=StreamingMode.RELAXED)
        c = ParallelPolicy(streaming_mode=StreamingMode.RELAXED)
        assert a != b == c
        # for code coverage
        assert a != "a"  # type: ignore[comparison-overlap]
        assert a.__eq__(123) is NotImplemented
        assert a.__ne__(123) is NotImplemented

    def test_partitioning_threshold(self) -> None:
        a = ParallelPolicy(
            partitioning_threshold=(TaskTarget.CPU, CPU_THRESHOLD)
        )
        b = ParallelPolicy()
        assert a != b
        assert (
            a.partitioning_threshold(TaskTarget.CPU)
            == CPU_THRESHOLD
            != b.partitioning_threshold(TaskTarget.CPU)
        )
        assert a.partitioning_threshold(
            TaskTarget.GPU
        ) == b.partitioning_threshold(TaskTarget.GPU)
        assert a.partitioning_threshold(
            TaskTarget.OMP
        ) == b.partitioning_threshold(TaskTarget.OMP)

        c = ParallelPolicy(
            partitioning_threshold={
                TaskTarget.CPU: CPU_THRESHOLD,
                TaskTarget.GPU: GPU_THRESHOLD,
                TaskTarget.OMP: OMP_THRESHOLD,
            }
        )
        assert (
            a.partitioning_threshold(TaskTarget.CPU)
            == c.partitioning_threshold(TaskTarget.CPU)
            == CPU_THRESHOLD
        )
        assert c.partitioning_threshold(TaskTarget.GPU) == GPU_THRESHOLD
        assert c.partitioning_threshold(TaskTarget.OMP) == OMP_THRESHOLD

    def test_set_partitioning_threshold(self) -> None:
        a = ParallelPolicy()
        a.set_partitioning_threshold(TaskTarget.CPU, CPU_THRESHOLD)
        assert a.partitioning_threshold(TaskTarget.CPU) == CPU_THRESHOLD
        a.set_partitioning_threshold(TaskTarget.GPU, GPU_THRESHOLD)
        assert a.partitioning_threshold(TaskTarget.GPU) == GPU_THRESHOLD
        a.set_partitioning_threshold(TaskTarget.OMP, OMP_THRESHOLD)
        assert a.partitioning_threshold(TaskTarget.OMP) == OMP_THRESHOLD


class TestParallelPolicyErrors:
    @pytest.mark.parametrize("factor", [-1, -2147483649])
    def test_overdecompose_negative(self, factor: int) -> None:
        msg = "can't convert negative value to uint32_t"
        with pytest.raises(OverflowError, match=msg):
            ParallelPolicy(overdecompose_factor=factor)

    def test_overdecompose_zero(self) -> None:
        msg = "overdecompose_factor must be 1 or more"
        with pytest.raises(ValueError, match=msg):
            ParallelPolicy(overdecompose_factor=0)

    def test_invalid_partitioning_threshold_type(self) -> None:
        msg = "an integer is required"
        with pytest.raises(TypeError, match=re.escape(msg)):
            ParallelPolicy(
                partitioning_threshold=("a", 3)  # type: ignore[arg-type]
            )

    def test_invalid_partitioning_threshold_tuple(self) -> None:
        msg = "partitioning_threshold must be a tuple of (TaskTarget, int)"
        with pytest.raises(ValueError, match=re.escape(msg)):
            ParallelPolicy(
                partitioning_threshold=()  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
