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

from argparse import (
    ArgumentParser,
    ArgumentTypeError,
    _ArgumentGroup as ArgumentGroup,
)
from dataclasses import dataclass, replace as dataclasses_replace
from typing import TYPE_CHECKING, Any

# This is imported here to re-export it
from legate.util.args import ArgSpec  # noqa: F401
from legate.util.args import Argument, Unset, entries

from .exception import LengthError

if TYPE_CHECKING:
    from ..cmake.cmake_flags import _CMakeVar


def _str_to_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    match v.casefold():
        case "yes" | "true" | "t" | "y" | "1":
            return True
        case "no" | "false" | "f" | "n" | "0" | "":
            return False
        case _:
            pass
    raise ArgumentTypeError(f"Boolean value expected, got '{v}'")


@dataclass(slots=True, frozen=True)
class ConfigArgument(Argument):
    cmake_var: _CMakeVar | None = None
    ephemeral: bool = False

    def add_to_argparser(self, parser: ArgumentParser | ArgumentGroup) -> None:
        r"""Add the contents of this ConfigArgument to an argument parser.

        Parameters
        ----------
        parser : ArgumentParser | ArgumentGroup
            The argument parser to add to.
        """
        spec = self.spec
        if spec.type is bool:
            to_replace: dict[str, Any] = {"type": _str_to_bool}

            def replace_if_unset(attr_name: str, value: Any) -> None:
                if getattr(spec, attr_name) is Unset:
                    to_replace[attr_name] = value

            replace_if_unset("nargs", "?")
            replace_if_unset("const", True)
            replace_if_unset("default", False)
            replace_if_unset("metavar", "bool")
            spec = dataclasses_replace(spec, **to_replace)

        kwargs = dict(entries(spec))
        parser.add_argument(self.name, **kwargs)


class ExclusiveArgumentGroup:
    def __init__(
        self, required: bool = False, **kwargs: ConfigArgument
    ) -> None:
        r"""Construct an ExclusiveArgumentGroup

        Parameters
        ----------
        required : bool, False
            Whether the argument group requires one of the arguments
            to be set.
        **kwargs : ConfigArgument
            The ConfigArgument's that make up this argument group.

        Raises
        ------
        LengthError
            If the number of arguments is less than 2 (since that would be
            pointless).
        TypeError
            If any of **kwargs is not a ConfigArgument.
        """
        if len(kwargs) < 2:
            raise LengthError(
                "Must supply at least 2 arguments to exclusive group"
            )
        self.group = kwargs
        self.required = required
        for attr, value in kwargs.items():
            if not isinstance(value, ConfigArgument):
                raise TypeError(f"Argument {attr} wrong type: {type(value)}")
            setattr(self, attr, value)
