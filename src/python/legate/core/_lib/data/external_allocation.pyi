# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

class ExternalAllocation:
    @staticmethod
    def from_sysmem(
        ptr: int, size: int, read_only: bool = True, source: object = None
    ) -> ExternalAllocation: ...
    @staticmethod
    def from_fbmem(
        device_id: int,
        ptr: int,
        size: int,
        read_only: bool = True,
        source: object = None,
    ) -> ExternalAllocation: ...
    @staticmethod
    def from_zcmem(
        ptr: int, size: int, read_only: bool = True, source: object = None
    ) -> ExternalAllocation: ...
    @staticmethod
    def from_dlpack(
        x: object,
        /,
        *,
        device: tuple[int, int] | None = None,
        copy: bool | None = None,
        read_only: bool | None = None,
    ) -> ExternalAllocation: ...
    @property
    def read_only(self) -> bool: ...
    @property
    def size(self) -> int: ...
