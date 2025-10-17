# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from legate.core import ParallelPolicy, Scope, StreamingMode

OVERDECOMPOSE_FACTOR = 42


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
