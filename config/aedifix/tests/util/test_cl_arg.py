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

import sys
from typing import TypeVar

import pytest

from ...util.cl_arg import CLArg

_T = TypeVar("_T")


class TestCLArg:
    @pytest.mark.parametrize("name", ("foo", "bar"))
    @pytest.mark.parametrize(
        "value,new_val",
        ((True, False), (1, -1), (2.0, 123123123.0), ("three", "four")),
    )
    @pytest.mark.parametrize("cl_set", (True, False))
    def test_create(
        self, name: str, value: _T, new_val: _T, cl_set: bool
    ) -> None:
        clarg = CLArg(name=name, value=value, cl_set=cl_set)
        assert clarg.name == name
        assert clarg.value == value
        assert clarg.cl_set == cl_set

        clarg.value = new_val
        assert clarg.name == name
        assert clarg.value == new_val
        assert clarg.cl_set is False

    @pytest.mark.parametrize("name", ("foo", "bar"))
    @pytest.mark.parametrize(
        "value,new_val",
        ((True, False), (1, -1), (2.0, 123123123.0), ("three", "four")),
    )
    @pytest.mark.parametrize("cl_set", (True, False))
    def test_eq(self, name: str, value: _T, new_val: _T, cl_set: bool) -> None:
        lhs = CLArg(name=name, value=value, cl_set=cl_set)
        rhs = CLArg(name=name, value=value, cl_set=cl_set)
        assert lhs == rhs


if __name__ == "__main__":
    sys.exit(pytest.main())
