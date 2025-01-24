#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import re
import sys
import json
import dataclasses
from argparse import ArgumentParser, FileType
from dataclasses import dataclass
from datetime import datetime, timedelta

_HELP_MSG = """
Process the log from a Legate testing run into a json file that can be
visualized using Chrome's built-in trace viewer. Different "processes" in the
visualization correspond to different shards (disjoint sets of GPUs), and
different "threads" to different tests. Only the first GPU-enabled test run in
the source file will be processed. Assumes the testing run was performed with
--debug.
"""

_INVOCATION_PAT = r"--gpu-bind ([0-9,]+) .* ([^ ]+\.py)"

_TIMESTAMP_PAT = r"[0-9][0-9]:[0-9][0-9]:[0-9][0-9]\.[0-9][0-9]"

_TEST_RESULT_PAT = (
    r"\[(PASS|FAIL|TIME)\] \(GPU\) ([0-9]+\.[0-9]+)s {("
    + _TIMESTAMP_PAT
    + r"), "
    + _TIMESTAMP_PAT
    + r"} ([^ ]+\.py)"
)

_FULL_RESULT_PAT = r"Results .* / ([0-9]+) files passed"


@dataclass
class Test:
    """Representation of a test invocation as an event on the Chrome trace
    viewer.
    """

    #: Title of event on the trace viewer; in our case the test filename
    name: str
    #: Process ID on the trace viewer; different "processes" correspond to
    #: different shards (disjoint sets of GPUs)
    pid: int
    #: Thread ID on the trace viewer; different "threads" correspond to
    #: different tests
    tid: int
    #: Start timestamp
    ts: int
    #: Duration of the test
    dur: int
    #: Label on the event box; used for test status (PASS/FAIL/TIME)
    args: dict[str, str]
    #: Event "phase"; must be set to "X" for standalone events
    ph: str = "X"


class LineParser:
    def __init__(self) -> None:
        self._state = self.BEFORE_GPU_STAGE
        self._gpu_for_test: dict[str, int] = {}
        self._next_tid = 0
        self._tests: list[Test] = []

    def _find_gpu(self, file: str) -> int:
        # We have to do it like this because the invocation and result lines
        # report the filename differently.
        for k, v in self._gpu_for_test.items():
            if k.endswith(file):
                return v
        msg = f"Invocation command not found for test {file}"
        raise ValueError(msg)

    def parse(self, line: str) -> None:
        self._state(line)

    def BEFORE_GPU_STAGE(self, line: str) -> None:
        if "Entering stage: GPU" in line:
            self._state = self.IN_GPU_STAGE

    def IN_GPU_STAGE(self, line: str) -> None:
        if "Exiting stage" in line:
            self._state = self.AFTER_GPU_STAGE
            return
        if (m := re.search(_INVOCATION_PAT, line)) is not None:
            gpu = int(m.group(1).split(",")[0])  # just keep the first GPU
            file = m.group(2)
            self._gpu_for_test[file] = gpu
            return
        if (m := re.search(_TEST_RESULT_PAT, line)) is not None:
            result = m.group(1)
            dur = int(float(m.group(2)) * 1000000)
            start = datetime.strptime(m.group(3), "%H:%M:%S.%f")
            ts = (start - datetime(1900, 1, 1)) // timedelta(microseconds=1)
            file = m.group(4)
            self._tests.append(
                Test(
                    name=result,
                    pid=self._find_gpu(file),
                    tid=self._next_tid,
                    ts=ts,
                    dur=dur,
                    args={"file": file},
                )
            )
            self._next_tid += 1

    def AFTER_GPU_STAGE(self, line: str) -> None:
        if (m := re.search(_FULL_RESULT_PAT, line)) is not None:
            expected = int(m.group(1))
            found = len(self._tests)
            if expected != found:
                msg = (
                    f"Expected to find {expected} invocations but only found "
                    f"{found}"
                )
                raise ValueError(msg)
            self._state = self.DONE

    def DONE(self, line: str) -> None:
        pass

    def sorted_tests(self) -> list[Test]:
        # First sort tests by start time, then duration.
        res = sorted(self._tests, key=lambda test: (test.ts, test.dur))
        for i, test in enumerate(res):
            test.tid = i
        return res


if __name__ == "__main__":
    arg_parser = ArgumentParser(description=_HELP_MSG)
    arg_parser.add_argument("input", type=FileType("r"), help="Input filename")
    arg_parser.add_argument(
        "output",
        nargs="?",
        type=FileType("w"),
        default=sys.stdout,
        help="Output filename; if not given print to stdout",
    )
    args = arg_parser.parse_args()

    line_parser = LineParser()
    for line in args.input:
        line_parser.parse(line)
    args.output.write(
        json.dumps(
            [dataclasses.asdict(test) for test in line_parser.sorted_tests()],
            indent=0,
        )
    )
