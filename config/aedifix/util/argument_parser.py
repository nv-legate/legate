# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from argparse import (
    Action,
    ArgumentParser,
    ArgumentTypeError,
    _ArgumentGroup as ArgumentGroup,
)
from dataclasses import (
    dataclass,
    fields as dataclasses_fields,
    replace as dataclasses_replace,
)
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

from .exception import LengthError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..cmake.cmake_flags import _CMakeVar


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


class Unset:
    pass


T = TypeVar("T")

NotRequired: TypeAlias = type[Unset] | T


@dataclass(slots=True, frozen=True)
class ArgSpec:
    dest: str
    action: NotRequired[ActionType] = Unset
    nargs: NotRequired[int | NargsType] = Unset
    const: NotRequired[Any] = Unset
    default: NotRequired[Any] = Unset
    type: NotRequired[type[Any] | Callable[[str], Any]] = Unset
    choices: NotRequired[Sequence[Any]] = Unset
    help: NotRequired[str] = Unset
    metavar: NotRequired[str] = Unset
    required: NotRequired[bool] = Unset

    def as_pruned_dict(self) -> dict[str, Any]:
        ret = {}
        for f in dataclasses_fields(self):
            name = f.name
            value = getattr(self, name)
            if value is not Unset:
                ret[name] = value
        return ret


@dataclass(slots=True, frozen=True)
class ConfigArgument:
    name: str
    """
    The command-line flag, e.g. --with-foo. This variable is pretty poorly
    named (no pun intended), it should really just be 'flag'.
    """

    spec: ArgSpec
    """The argparse argument spec."""

    cmake_var: _CMakeVar | None = None
    """The CMake variable corresponding to this variable"""

    ephemeral: bool = False
    """
    Whether the flag should be stored in the reconfigure script. An ephemeral
    flag (when this is True) will NOT be stored in the reconfigure script.
    This is used, for example, for the --with-clean flag, since it should
    only happen once. Running reconfigure should not delete the arch directory
    over and over again.
    """

    enables_package: bool = False
    """
    Whether this flag 'enables' the package in question. If True, and there is
    a truthy value give on the command-line for the flag, then the package will
    consider itself enabled. This is commonly used with the --with-foo-dir
    class of flags. --with-foo-dir implies --with-foo, so the user shouldn't
    have to pass both.
    """

    primary: bool = False
    """
    Whether this flag is the 'primary' enabler or disabler of the package. The
    primary flags is the ultimate tie-breaker in deciding if a package is
    enabled or disabled. The rationale is that --with-cuda-dir=path
    --with-cudac=nvcc --with-cuda=0 should always result in CUDA being disabled
    even thoough all the other flags are truthy.
    """

    @staticmethod
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
        msg = f"Boolean value expected, got '{v}'"
        raise ArgumentTypeError(msg)

    def add_to_argparser(self, parser: ArgumentParser | ArgumentGroup) -> None:
        r"""Add the contents of this ConfigArgument to an argument parser.

        Parameters
        ----------
        parser : ArgumentParser | ArgumentGroup
            The argument parser to add to.
        """
        spec = self.spec
        if spec.type is bool:
            to_replace: dict[str, Any] = {"type": self._str_to_bool}

            def replace_if_unset(attr_name: str, value: Any) -> None:
                if getattr(spec, attr_name) is Unset:
                    to_replace[attr_name] = value

            replace_if_unset("nargs", "?")
            replace_if_unset("const", True)
            replace_if_unset("default", False)
            replace_if_unset("metavar", "bool")
            spec = dataclasses_replace(spec, **to_replace)

        kwargs = spec.as_pruned_dict()
        parser.add_argument(self.name, **kwargs)


class ExclusiveArgumentGroup:
    def __init__(
        self, *, required: bool = False, **kwargs: ConfigArgument
    ) -> None:
        r"""Construct an ExclusiveArgumentGroup.

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
        if len(kwargs) < 2:  # noqa: PLR2004
            msg = "Must supply at least 2 arguments to exclusive group"
            raise LengthError(msg)
        self.group = kwargs
        self.required = required
        for attr, value in kwargs.items():
            if not isinstance(value, ConfigArgument):
                # Obviously this _should_ be unreachable, but bugs happen all
                # the time :)
                msg = f"Argument {attr} wrong type: {type(value)}"  # type: ignore[unreachable]
                raise TypeError(msg)
            setattr(self, attr, value)
