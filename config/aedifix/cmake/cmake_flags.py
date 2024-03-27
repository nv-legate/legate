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

import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from shlex import quote as shlex_quote
from types import GeneratorType
from typing import Any, Final, Literal, TypeVar, cast as TYPE_CAST

_T = TypeVar("_T")


class CMakeFlagBase(ABC):
    __slots__ = "_name", "_value", "_prefix", "_type"

    _name: Final[str]
    _prefix: Final[str]
    _type: Final[str]

    def __init__(
        self,
        name: str,
        value: _T | None = None,
        prefix: str = "-D",
        type_str: str = "STRING",
    ) -> None:
        r"""Construct a CMakeFlagBase

        Parameters
        ----------
        name : str
            The name of the CMake variable.
        value : Any, optional
            The initial value for the variable.
        prefix : str, '-D'
            The command line prefix for the variable.
        type_str : str, optional
            The type string of the cmake variable
        """
        super().__init__()
        # Init these first in case derived classes want to inspect them in
        # _sanitize_value()
        self._name = name
        self._prefix = prefix
        self._type = type_str

        if value is not None:
            value = self._sanitize_value(value)

        self._value = value

    @property
    def name(self) -> str:
        r"""Get the name of the CMake variable.

        Returns
        -------
        name : str
            The name of the variable, e.g. 'CMAKE_C_FLAGS'.
        """
        return self._name

    @property
    def prefix(self) -> str:
        r"""Get the prefix of the CMake variable.

        Returns
        -------
        prefix : str
            The prefix of the variable, e.g. '-D'.
        """
        return self._prefix

    @property
    def type(self) -> str:
        r"""Get the CMake type string of the variable.

        Returns
        -------
        type : str
            The CMake type, e.g. 'BOOL'.
        """
        return self._type

    @property
    def value(self) -> Any:
        r"""Get the value of the CMake variable.

        Returns
        -------
        value : Any
            The value of the variable, e.g. 'ON'.
        """
        return self._value

    @value.setter
    def value(self, val: _T) -> None:
        self._value = self._sanitize_value(val)

    @abstractmethod
    def _sanitize_value(self, val: _T) -> Any:
        r"""The callback hook for value setter, which must be overriden by
        derived classes.

        Parameters
        ----------
        val : Any
            The value to assign to `self._value`.

        Returns
        -------
        val : SomeType
            The sanitized value.

        Raises
        ------
        TypeError
            If the input type could not be properly sanitized.
        ValueError
            If the input value could not be properly sanitized.

        Notes
        -----
        Derived classes must return a concrete value from this function. Any
        unhandled types *must* raise a TypeError, and any failure to sanitized
        handled types *must* raise a ValueError.
        """
        raise NotImplementedError()

    def canonicalize(self) -> CMakeFlagBase | None:
        r"""Canonicalize the CMake variable.

        Returns
        -------
        canonical : CMakeFlagBase | None
            The canonical form of the CMake variable, or None if the variable
            is not canonicalizeable.
        """
        valid, val = self._canonicalize_cb()
        if valid:
            # type(self) is critical, we want to construct the most derived
            # type here.
            return type(self)(self.name, val, self.prefix)
        return None

    def _canonicalize_cb(self) -> tuple[bool, Any | None]:
        r"""Callback to construct the canonical object. Must be implemented
        by the derived class.

        Returns
        -------
        valid : bool
            True if `val` is a valid, canonical value for this CMake variable,
            False otherwise.
        val : Any
            The canonical value of the variable, e.g. 'ON' or ['-O2', '-g3'].
        """
        val = self.value
        return (val is not None), val

    def to_command_line(self, quote: bool = False) -> str:
        r"""Create a command line friendly representation of the CMake
        variable.

        Parameters
        ----------
        quote : bool, False
            True if the value should be quoted (if necessary), False otherwise.

        Returns
        -------
        value : str
            The command line form of the variable and its values.

        Raises
        ------
        ValueError
            If the variable is empty

        Notes
        -----
        If the variable is to be passed to `subprocess`, then `quote` should
        almost certainly be false, since commands taken in list form are
        automatically treated as quoted already.
        """
        val = self.value
        if val is None:
            raise ValueError(
                f'Cannot convert "{self.name}" to command-line, '
                "have empty value"
            )
        if quote:
            val = shlex_quote(str(val))
        return f"{self.prefix}{self.name}:{self.type}={val}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            (self.name == other.name)
            and (self.prefix == other.prefix)
            and (self.type == other.type)
            and (self.value == other.value)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, "
            f"prefix={self.prefix}, "
            f"type={self.type}, "
            f"value={self.value}"
            ")"
        )


class CMakeList(CMakeFlagBase):
    def __init__(
        self, name: str, value: Sequence[Any] | None = None, prefix: str = "-D"
    ) -> None:
        super().__init__(name=name, value=value, prefix=prefix)

    @staticmethod
    def _sanitize_value(value: Any) -> list[_T]:
        if isinstance(value, (list, tuple, GeneratorType)):
            return list(value)
        raise TypeError(type(value))

    def _canonicalize_cb(self) -> tuple[bool, list[str] | None]:
        if (val := self.value) is not None:
            val = [v for v in map(lambda x: str(x).strip(), val) if v]
        return bool(val), val

    def to_command_line(self, quote: bool = False) -> str:
        if (val := self.value) is None:
            val = []
        val = " ".join(map(str, val))
        if quote:
            val = shlex_quote(val)
        return f"{self.prefix}{self.name}:{self.type}={val}"


class CMakeBool(CMakeFlagBase):
    def __init__(
        self,
        name: str,
        value: bool | int | str | None = None,
        prefix: str = "-D",
    ) -> None:
        super().__init__(
            name=name, value=value, prefix=prefix, type_str="BOOL"
        )

    @staticmethod
    def _sanitize_value(value: Any) -> Literal["ON", "OFF"]:
        if isinstance(value, str):
            if value not in {"OFF", "ON"}:
                raise ValueError(f"value: {value} not in [OFF, ON]")
            # mypy (as of 1.8.0)  does not yet narrow the type of value from
            # str to Literal["ON", "OFF"] based on the 'in' check above (see
            # https://github.com/python/mypy/issues/12535), so we have to tell
            # it.
            return TYPE_CAST(Literal["ON", "OFF"], value)
        if isinstance(value, (bool, int)):
            if isinstance(value, int) and value not in {0, 1}:
                raise ValueError(f"value: {value} not in [0, 1]")
            return "ON" if value else "OFF"
        raise TypeError(type(value))


class CMakeInt(CMakeFlagBase):
    def __init__(
        self,
        name: str,
        value: int | bool | str | float | None = None,
        prefix: str = "-D",
    ) -> None:
        super().__init__(name=name, value=value, prefix=prefix)

    @staticmethod
    def _sanitize_value(value: Any) -> int:
        if isinstance(value, (bool, str, float, int)):
            return int(value)
        raise TypeError(type(value))


class CMakeString(CMakeFlagBase):
    def __init__(
        self, name: str, value: str | None = None, prefix: str = "-D"
    ) -> None:
        super().__init__(name=name, value=value, prefix=prefix)

    @staticmethod
    def _sanitize_value(value: Any) -> str:
        if isinstance(value, str):
            return value
        raise TypeError(type(value))


class CMakePath(CMakeFlagBase):
    def __init__(
        self, name: str, value: str | Path | None = None, prefix: str = "-D"
    ) -> None:
        super().__init__(name=name, value=value, prefix=prefix)
        if self.value is not None:
            self.__update_type(self.value)

    def __update_type(self, value: Path) -> None:
        if value.exists():
            # We are re-assigning a Final value here, but this class cannot
            # work without doing so.
            self._type = (  # type: ignore[misc]
                "PATH" if value.is_dir() else "FILEPATH"
            )

    def _sanitize_value(self, value: Any) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(type(value))
        value = Path(value).resolve()
        self.__update_type(value)
        return value


class CMakeExecutable(CMakeFlagBase):
    def __init__(
        self, name: str, value: str | Path | None = None, prefix: str = "-D"
    ) -> None:
        super().__init__(
            name=name, value=value, prefix=prefix, type_str="FILEPATH"
        )

    @staticmethod
    def _sanitize_value(value: Any) -> Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(type(value))
        if not isinstance(value, Path):
            value = Path(value)
        if value.exists():
            if value.is_dir():
                raise ValueError(f"Got a directory as an executable: {value}")
        elif valtmp := shutil.which(value):
            value = Path(valtmp)
        return value


class _CMakeVar(str):
    def _set_cmake_type(
        self,
        ty: type[CMakeFlagBase],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self.__cmake_type = ty
        self.__cmake_type_args = args
        self.__cmake_type_kwargs = kwargs

    def __config_cmake_type__(self) -> CMakeFlagBase:
        return self.__cmake_type(
            self, *self.__cmake_type_args, **self.__cmake_type_kwargs
        )


def CMAKE_VARIABLE(
    name: str, ty: type[CMakeFlagBase], *args: Any, **kwargs: Any
) -> _CMakeVar:
    ret = _CMakeVar(name)
    ret._set_cmake_type(ty, args, kwargs)
    return ret
