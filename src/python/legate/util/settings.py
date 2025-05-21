# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Control global configuration options with environment variables.

Precedence
~~~~~~~~~~

Setting values are always looked up in the following prescribed order:

immediately supplied values
    These are values that are passed to the setting:

    .. code-block:: python

        settings.consensus(value)

    If ``value`` is not None, then it will be returned, as-is. Otherwise, if
    None is passed, then the setting will continue to look down the search
    order for a value. This is useful for passing optional function parameters
    that are None by default. If the parameter is passed to the function, then
    it will take precedence.

previously user-set values
    If the value is set explicitly in code:

    .. code-block:: python

        settings.minified = False

    Then this value will take precedence over other sources. Applications
    may use this ability to set values supplied on the command line, so that
    they take precedence over environment variables.

environment variable
    Values found in the associated environment variables:

    .. code-block:: sh

        LEGATE_CONSENSUS=1 legate script.py

local defaults
    These are default values defined when accessing the setting:

    .. code-block:: python

        settings.consensus(default=True)

    Local defaults have lower precedence than every other setting mechanism
    except global defaults.

global defaults
    These are default values defined by the setting declarations. They have
    lower precedence than every other setting mechanism.

If no value is obtained after searching all of these locations, then a
RuntimeError will be raised.

"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    cast as typing_cast,
)

__all__ = (
    "PrioritizedSetting",
    "Settings",
    "convert_bool",
    "convert_int",
    "convert_str",
    "convert_str_seq",
)


class _Unset:
    pass


T = TypeVar("T", covariant=True)  # noqa: PLC0105

Unset: TypeAlias = T | type[_Unset]


def convert_str(value: str) -> str:
    """Return a string as-is."""
    return value


def convert_int(value: str) -> int:
    """Return an integer value."""
    return int(value)


def convert_bool(value: bool | str) -> bool:
    """Convert a string to True or False.

    If a boolean is passed in, it is returned as-is. Otherwise the function
    maps the strings "0" -> False and "1" -> True.

    Parameters
    ----------
    value : bool | str
        A string value to convert to bool

    Returns
    -------
    The converted boolean

    Raises
    ------
    ValueError
        If the input could not be converted
    """
    if isinstance(value, bool):
        return value

    match value.casefold():
        case "1":
            return True
        case "0" | "":
            return False

    msg = f'Cannot convert {value!r} to bool, use "0" or "1"'
    raise ValueError(msg)


def convert_str_seq(
    value: list[str] | tuple[str, ...] | str,
) -> tuple[str, ...]:
    """Convert a string to a list of strings.

    If a list or tuple is passed in, it is returned as-is.

    Args:
        value (seq[str] or str) :
            A string to convert to a list of strings

    Returns
    -------
        list[str]

    Raises
    ------
        ValueError

    """
    if isinstance(value, (list, tuple)):
        return tuple(value)

    try:
        return tuple(value.split(","))
    except Exception as e:
        msg = f"Cannot convert {value} to list value"
        raise ValueError(msg) from e


class ConversionFnWithType(Generic[T], Protocol):
    type: str

    def __call__(self, value: Any) -> T: ...


ConversionFn: TypeAlias = Callable[[Any], T]

Converter: TypeAlias = ConversionFn[T] | ConversionFnWithType[T]


class SettingBase(Generic[T]):
    def __init__(
        self,
        name: str,
        default: Unset[T] = _Unset,
        convert: Converter[T] | None = None,
        help: str = "",  # noqa: A002
    ) -> None:
        self._default = default
        self._convert = (
            convert if convert else typing_cast(Converter[T], convert_str)
        )
        self._help = help
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def help(self) -> str:
        return self._help

    @property
    def default(self) -> Unset[T]:
        return self._default

    @property
    def convert_type(self) -> str:
        if self._convert is convert_str:
            return "str"
        if self._convert is convert_int:
            return "int"
        if self._convert is convert_bool:
            return 'bool ("0" or "1")'
        if self._convert is convert_str_seq:
            return "tuple[str, ...]"
        if callable(self._convert):
            convert = typing_cast(ConversionFnWithType[T], self._convert)
            if hasattr(convert, "type"):
                return convert.type
        msg = "unreachable"
        raise RuntimeError(msg)


class PrioritizedSetting(SettingBase[T]):
    """Return a value for a global setting according to configuration
    precedence.

    The following methods are searched in order for the setting:

    4. immediately supplied values
    3. previously user-set values (e.g. set from command line)
    2. environment variable
    1. local defaults
    0. global defaults

    If a value cannot be determined, a RuntimeError is raised.

    The ``env_var`` argument specifies the name of an environment to check for
    setting values, e.g. ``"LEGATE_CHECK_CYCLE"``.

    The optional ``default`` argument specified an implicit default value for
    the setting that is returned if no other methods provide a value.

    A ``convert`` argument may be provided to convert values before they are
    returned.
    """

    _user_value: Unset[str | T]

    def __init__(
        self,
        name: str,
        env_var: str | None = None,
        default: Unset[T] = _Unset,
        convert: Converter[T] | None = None,
        help: str = "",  # noqa: A002
    ) -> None:
        super().__init__(name, default, convert, help)
        self._env_var = env_var
        self._user_value = _Unset

    def __call__(
        self, value: T | str | None = None, default: Unset[T] = _Unset
    ) -> T:
        """Return the setting value according to the standard precedence.

        Args:
            value (any, optional):
                An optional immediate value. If not None, the value will
                be converted, then returned.

            default (any, optional):
                An optional default value that only takes precedence over
                implicit default values specified on the property itself.

        Returns
        -------
            str or int or float

        Raises
        ------
            RuntimeError
        """
        # 4. immediate values
        if value is not None:
            return self._convert(value)

        # 3. previously user-set value
        if self._user_value is not _Unset:
            return self._convert(self._user_value)

        # 2. environment variable
        if self._env_var and self._env_var in os.environ:
            return self._convert(os.environ[self._env_var])

        # 1. local defaults
        if default is not _Unset:
            return self._convert(default)

        # 0. global defaults
        if self._default is not _Unset:
            return self._convert(self._default)

        msg = f"No configured value found for setting {self._name!r}"
        raise RuntimeError(msg)

    def __get__(  # noqa: D105
        self, instance: Any, owner: type[Any]
    ) -> PrioritizedSetting[T]:
        return self

    def __set__(self, instance: Any, value: str | T) -> None:  # noqa: D105
        self.set_value(value)

    def set_value(self, value: str | T) -> None:
        """Specify a value for this setting programmatically.

        A value set this way takes precedence over all other methods except
        immediate values.

        Args:
            value (str or int or float):
                A user-set value for this setting

        Returns
        -------
            None
        """
        # It is usually not advised to store any data directly on descriptors,
        # since they are shared by all instances. But in our case we only ever
        # have a single instance of a given settings object.
        self._user_value = value

    def unset_value(self) -> None:
        """Unset the previous user value such that the priority is reset."""
        self._user_value = _Unset

    @property
    def env_var(self) -> str | None:  # noqa: D102
        return self._env_var


class EnvOnlySetting(SettingBase[T]):
    """Return a value for a global environment variable setting. The value
    is cached upon first read (i.e. it will not update if the value of the
    environment variable subsequently changes).

    A ``convert`` argument may be provided to convert values before they are
    returned.
    """

    _cached = False

    _cached_value: T

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        env_var: str,
        default: Unset[T] = _Unset,
        test_default: Unset[T] = _Unset,
        convert: Any | None = None,
        help: str = "",  # noqa: A002
    ) -> None:
        super().__init__(name, default, convert, help)
        self._test_default = test_default
        self._env_var = env_var

    def __call__(self) -> T:
        if self._cached:
            return self._cached_value

        if self._env_var in os.environ:
            self._cached_value = self._convert(os.environ[self._env_var])

        else:
            # unfortunate necessity for LEGATE_TEST
            test = convert_bool(os.environ.get("LEGATE_TEST", ""))
            if test and self.test_default is not _Unset:
                self._cached_value = self._convert(self.test_default)

            elif self._default is not _Unset:
                self._cached_value = self._convert(self.default)

            else:
                msg = (
                    f"EnvOnlySetting {self.name}: env var {self._env_var} "
                    "is not set and no default has been provided"
                )
                raise ValueError(msg)

        self._cached = True
        return self._cached_value

    def __get__(self, instance: Any, owner: type[Any]) -> EnvOnlySetting[T]:
        return self

    @property
    def env_var(self) -> str | None:
        return self._env_var

    @property
    def test_default(self) -> Unset[T]:
        return self._test_default


class Settings:
    pass
