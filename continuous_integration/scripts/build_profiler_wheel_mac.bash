#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export MACOSX_DEPLOYMENT_TARGET=11.0
export WHEEL_DIR=${WHEEL_DIR:-/tmp/profiler_out}
export WHEEL_TEST_DIR=${WHEEL_TEST_DIR:-/tmp/wheel_test}

get_legate_version() {
  if [[ -n "${PROFILER_VERSION:-}" ]]; then
    echo "${PROFILER_VERSION}"
    return
  fi

  python - <<'PY'
import os
from pathlib import Path

from setuptools_scm import get_version

repo_dir = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()
print(
    get_version(
        root=repo_dir,
        version_scheme="guess-next-dev",
        local_scheme="node-and-date",
    )
)
PY
}

build_wheel_with_maturin() {
  echo "Build the profiler wheel package"
  RUST_TARGET="aarch64-apple-darwin"

  # shellcheck disable=SC2154
  PKG_DIR="${GITHUB_WORKSPACE}/scripts/build/python/legate-profiler"
  # shellcheck disable=SC2154
  LEGION_VERSION_FILE="${GITHUB_WORKSPACE}/src/cmake/versions/legion_version.json"

  LEGION_GIT_URL=$(jq -r '.packages.Legion.git_url' "${LEGION_VERSION_FILE}")
  echo "  LEGION_GIT_URL: ${LEGION_GIT_URL}"
  LEGION_GIT_REV=$(jq -r '.packages.Legion.git_tag' "${LEGION_VERSION_FILE}")
  echo "  LEGION_GIT_REV: ${LEGION_GIT_REV}"

  git clone "${LEGION_GIT_URL}" legion
  git -C legion checkout "${LEGION_GIT_REV}"

  PROF_RS_DIR="$(realpath legion/tools/legion_prof_rs)"

  echo "Building profiler.. "
  rm -rf "${WHEEL_DIR}"
  mkdir -p "${WHEEL_DIR}"
  cp "${PKG_DIR}/pyproject.toml" "${PROF_RS_DIR}/pyproject.toml"

  python -m pip install --upgrade pip maturin setuptools_scm

  # Replace existing legion version with legate version
  LEGATE_VERSION=$(get_legate_version)
  sed -E -i.bak 's/^version = "[^"]*"$/version = "'"${LEGATE_VERSION}"'"/' "${PROF_RS_DIR}/pyproject.toml"
  rm -f "${PROF_RS_DIR}/pyproject.toml.bak"
  ls -lAt "${PROF_RS_DIR}"
  cat "${PROF_RS_DIR}/pyproject.toml"

  cd "${PROF_RS_DIR}"
  maturin build \
    --manifest-path "${PROF_RS_DIR}/Cargo.toml" \
    --all-features \
    --release \
    --out "${WHEEL_DIR}" \
    --target "${RUST_TARGET}"
}

verify_profile_package() {
  echo "Verify the profiler wheel package"

  if ! ls "${WHEEL_DIR}"/*.whl >/dev/null 2>&1; then
    echo "Error: No wheel found in ${WHEEL_DIR}"
    exit 1
  fi

  rm -rf "${WHEEL_TEST_DIR}"
  mkdir -p "${WHEEL_TEST_DIR}"

  python -m venv "${WHEEL_TEST_DIR}"
  "${WHEEL_TEST_DIR}/bin/python" -m pip install --upgrade pip
  "${WHEEL_TEST_DIR}/bin/python" -m pip install "${WHEEL_DIR}"/*.whl

  legion_prof_path="${WHEEL_TEST_DIR}/bin/legion_prof"
  if [[ ! -x "${legion_prof_path}" ]]; then
    echo "Error: legion_prof binary not found.."
    exit 1
  fi
  echo "legion_prof path: ${legion_prof_path}"

  # Basic verification
  "${legion_prof_path}" --help
  "${legion_prof_path}" -V
  echo "Verification done"
}

main() {
  local op="${1:-all}"

  case "${op}" in
    build)
      build_wheel_with_maturin
      ;;
    verify)
      verify_profile_package
      ;;
    *)
      echo "Usage: $0 [build|verify]"
      exit 1
      ;;
  esac

  echo "Done"
}

main "$@"
