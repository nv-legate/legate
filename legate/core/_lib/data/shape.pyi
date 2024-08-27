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

from collections.abc import Collection, Iterator

class Shape:
    def __init__(self, value: Shape | Collection[int]) -> None: ...
    @property
    def extents(self) -> tuple[int, ...]: ...
    @property
    def volume(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    def __getitem__(self, idx: int) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
