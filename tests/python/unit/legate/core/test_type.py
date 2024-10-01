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

from typing import Any

import numpy as np
import pytest

from legate import Type, types as ty


class TestType:
    @pytest.mark.parametrize(
        "obj,expected_ty",
        [
            (0, ty.uint8),
            (1, ty.uint8),
            (-1, ty.int8),
            (256, ty.uint16),
            (-129, ty.int16),
            (65536, ty.uint32),
            (-32769, ty.int32),
            (4_294_967_296, ty.uint64),
            (-2_147_483_649, ty.int64),
            (True, ty.bool_),
            (False, ty.bool_),
            (None, ty.null_type),
            ("foo", ty.string_type),
            # The following always take the largest possible floating point
            # type in order to guarantee the value round-trips:
            # float(ty.floating_point(fp)) == fp must hold true.
            (1.23, ty.float64),
            (complex(1, 2), ty.complex128),
            (b"foo", ty.binary_type(3)),
            ([1, 2, 3], ty.array_type(ty.uint8, 3)),
            ((1, 2, 3), ty.array_type(ty.uint8, 3)),
            (
                np.array([1.0, 2.0, 3.0]),
                ty.array_type(ty.float64, 3),
            ),
            (
                np.array([1.0, 2.0, 3.0], dtype=np.float32),
                ty.array_type(ty.float32, 3),
            ),
            (np.array([None, None, None]), ty.array_type(ty.null_type, 3)),
            (np.array(1, dtype=np.int8), ty.array_type(ty.int8, 1)),
        ],
    )
    def test_from_python_type(self, obj: Any, expected_ty: Type) -> None:
        cur_ty = Type.from_python_object(obj)
        assert cur_ty == expected_ty


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(sys.argv))
