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

from importlib import reload

from pytest_mock import MockerFixture

import legate.driver.defaults as m


def test_LEGATE_NODES() -> None:
    assert m.LEGATE_NODES == 1


def test_LEGATE_RANKS_PER_NODE() -> None:
    assert m.LEGATE_RANKS_PER_NODE == 1


def test_LEGATE_LOG_DIR(mocker: MockerFixture) -> None:
    mocker.patch("os.getcwd", return_value="foo")
    reload(m)
    assert m.LEGATE_LOG_DIR == "foo"
