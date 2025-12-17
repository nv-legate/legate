# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from legate.util import info


def test_legion_version_is_recorded() -> None:
    legion_version = info.package_versions()["legion"]

    assert legion_version not in ("", info.FAILED_TO_DETECT)
    assert not legion_version.startswith("(")
