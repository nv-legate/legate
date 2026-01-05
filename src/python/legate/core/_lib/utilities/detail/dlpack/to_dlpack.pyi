# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing_extensions import CapsuleType

from ....data.physical_store import PhysicalStore

def to_dlpack(
    store: PhysicalStore,
    *,
    stream: int | None = None,
    max_version: tuple[int, int] | None = None,
    dl_device: tuple[int, int] | None = None,
    copy: bool | None = None,
) -> CapsuleType: ...
