# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Generic, TypeVar

_T = TypeVar("_T")


class CLArg(Generic[_T]):
    __slots__ = "_cl_set", "_name", "_value"

    def __init__(self, name: str, value: _T | None, *, cl_set: bool) -> None:
        r"""Construct a ``CLArg``.

        Parameters
        ----------
        name : str
            The name of the command line argument.
        value : T
            The value of the command line argument.
        cl_set : bool
            True if the value was set by the user on the command line, False
            otherwise.
        """
        self._name = name
        self._value = value
        self._cl_set = cl_set

    @property
    def name(self) -> str:
        r"""Get the name of a command line argument.

        Returns
        -------
        name : str
            The name of the command line argument.
        """
        return self._name

    @property
    def value(self) -> _T | None:
        r"""Get the value of the command line argument.

        Returns
        -------
        value : T
            The value of the command line argument.
        """
        return self._value

    @value.setter
    def value(self, val: _T) -> None:
        r"""Set the value of a command line argument.

        Parameters
        ----------
        val : T
            The new value.
        """
        self._value = val
        self._cl_set = False

    @property
    def cl_set(self) -> bool:
        r"""Get whether the value was set by the user on command line.

        Returns
        -------
        set : bool
            True if set by the user, false otherwise.
        """
        return self._cl_set

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CLArg):
            return NotImplemented
        return (
            (self.name == other.name)
            and (self.value == other.value)
            and (self.cl_set == other.cl_set)
        )

    def __repr__(self) -> str:
        return (
            "CLArg("
            f"name={self.name}, "
            f"value={self.value}, "
            f"cl_set={self.cl_set}"
            ")"
        )
