# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import json

import pytest

from legate import Scope, track_provenance


@track_provenance()
def func() -> str:
    return Scope.provenance()


@track_provenance()
def unnested() -> str:
    return func()


@track_provenance(nested=True)
def nested() -> str:
    return func()


class Test_track_provenance:
    def test_unnested(self) -> None:
        human, machine = json.loads(unnested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine

    def test_nested(self) -> None:
        human, machine = json.loads(nested())
        assert "test_runtime.py" in human
        assert "test_runtime.py" in machine["file"]
        assert "line" in machine


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
