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

from argparse import Action
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, TypeAlias, TypeVar


class _UnsetType:
    pass


Unset = _UnsetType()


T = TypeVar("T")

NotRequired: TypeAlias = _UnsetType | T


# https://docs.python.org/3/library/argparse.html#action
ActionType: TypeAlias = (
    Literal[
        "store",
        "store_const",
        "store_true",
        "append",
        "append_const",
        "count",
        "help",
        "version",
        "extend",
    ]
    | type[Action]
)

# https://docs.python.org/3/library/argparse.html#nargs
NargsType: TypeAlias = Literal["?", "*", "+", "..."]


@dataclass(frozen=True)
class ArgSpec:
    dest: str
    action: NotRequired[ActionType] = Unset
    nargs: NotRequired[int | NargsType] = Unset
    const: NotRequired[Any] = Unset
    default: NotRequired[Any] = Unset
    type: NotRequired[type[Any]] = Unset
    choices: NotRequired[Sequence[Any]] = Unset
    help: NotRequired[str] = Unset
    metavar: NotRequired[str] = Unset
    required: NotRequired[bool] = Unset


@dataclass(frozen=True)
class Argument:
    name: str
    spec: ArgSpec

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(entries(self.spec))


def entries(obj: Any) -> Iterable[tuple[str, Any]]:
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is not Unset:
            yield (f.name, value)
