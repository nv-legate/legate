# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from argparse import SUPPRESS, Action, ArgumentParser, Namespace
from dataclasses import dataclass, fields
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    NoReturn,
    TypeAlias,
    TypeVar,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


__all__ = (
    "ActionType",
    "ArgSpec",
    "ExtendAction",
    "InfoAction",
    "MultipleChoices",
    "NargsType",
    "NotRequired",
    "Unset",
)


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
        return dict(_entries(self.spec))


def _entries(obj: Any) -> Iterable[tuple[str, Any]]:
    for f in fields(obj):
        value = getattr(obj, f.name)
        if value is not Unset:
            yield (f.name, value)


class MultipleChoices(Generic[T]):
    """A container that reports True for any item or subset inclusion.

    Parameters
    ----------
    choices: Iterable[T]
        The values to populate the container.

    Examples
    --------
    >>> choices = MultipleChoices(["a", "b", "c"])

    >>> "a" in choices
    True

    >>> ("b", "c") in choices
    True

    """

    def __init__(self, choices: Iterable[T]) -> None:
        self._choices = set(choices)

    def __contains__(self, x: T | Sequence[T]) -> bool:  # noqa: D105
        if isinstance(x, (list, tuple)):
            return set(x).issubset(self._choices)
        return x in self._choices

    def __iter__(self) -> Iterator[T]:  # noqa: D105
        return self._choices.__iter__()


class ExtendAction(Action, Generic[T]):
    """A custom argparse action to collect multiple values into a list."""

    def __call__(  # noqa: D102
        self,
        parser: ArgumentParser,  # noqa: ARG002
        namespace: Namespace,
        values: str | Sequence[T] | None,
        option_string: str | None = None,  # noqa: ARG002
    ) -> None:
        items = getattr(namespace, self.dest) or []
        if isinstance(values, (list, tuple)):
            items.extend(values)
        else:
            items.append(values)
        # removing any duplicates before storing
        setattr(namespace, self.dest, list(set(items)))


class InfoAction(Action):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["metavar"] = None
        kwargs["nargs"] = 0
        kwargs["default"] = SUPPRESS
        super().__init__(*args, **kwargs)

    def __call__(  # noqa: D102
        self,
        parser: ArgumentParser,  # noqa: ARG002
        namespace: Namespace,  # noqa: ARG002
        values: str | Sequence[T] | None,  # noqa: ARG002
        option_string: str | None = None,  # noqa: ARG002
    ) -> NoReturn:
        from .info import print_build_info

        print_build_info()
        sys.exit()
