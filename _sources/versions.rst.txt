..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

Versions and Tags
=================

Overview
--------

Summary of the versioning and release methodology used by Legate projects.

Versioning method
-----------------

All Legate projects use the `CalVer`_ versioning method for all releases
starting with June 2021 release.

The Legate team is aware of the impacts that public API changes cause to users,
so API & ABI compatibility is guaranteed within each `YY.MM` version.

Release types and tagging
-------------------------

Using CalVer for versioning, Legate projects use the notation `YY.MM.PP` for
releases/tags, where `YY` indicates the zero padded year, `MM` indicates the zero
padded month, and `PP` indicates the zero padded hotfix/patch version. Each
release is accompanied by a tag in the git repo with the same formatting and
leading `v`.

Hotfix/Patch
____________

A hotfix/patch release occurs for the current release, incrementing the
hotfix/patch version number by 1.

There is no limit or time constraint of these releases as they are governed by
the need to fix critical issues in the current release. Generally, hotfix/patch
releases contain only one change and are typically bug fixes; new features
should not be introduced in this way.

Weekly releases
_______________

Lightly validated packages are uploaded to the main `legate` conda channel at a
roughly weekly cadence. These packages are versioned as `YY.MM.PP.devXXX`, where
`XXX` is a monotonically increasing number. These packages do not go through
full QA validation, and thus may contain functional and performance regressions.

Nightly builds
______________

Nightly builds are uploaded to a separate channel, `legate-nightly`. Please be
aware that these packages are not validated at all.

.. _CalVer: https://calver.org/
