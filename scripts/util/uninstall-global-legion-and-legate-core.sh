#! /usr/bin/env bash

if [[ -z ${CONDA_PREFIX} ]]; then
       exit 1
fi

rm -rf "$(find "${CONDA_PREFIX}" -mindepth 1 -type d -name '*realm*' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type d -name '*legion*' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type d -name '*legate*' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type d -name '*Legion*' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type f -name 'realm*.h' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type f -name 'legion*.h' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type f -name 'pygion.py' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type f -name 'legion_top.py' || true)" \
       "$(find "${CONDA_PREFIX}" -mindepth 1 -type f -name 'legion_cffi.py' || true)" \
       "$(find "${CONDA_PREFIX}/lib" -mindepth 1 -type f -name 'librealm*' || true)" \
       "$(find "${CONDA_PREFIX}/lib" -mindepth 1 -type f -name 'libregent*' || true)" \
       "$(find "${CONDA_PREFIX}/lib" -mindepth 1 -type f -name 'liblegion*' || true)" \
       "$(find "${CONDA_PREFIX}/lib" -mindepth 1 -type f -name 'liblgcore*' || true)" \
       "$(find "${CONDA_PREFIX}/lib" -mindepth 1 -type f -name 'legate.core.egg-link' || true)" \
       "$(find "${CONDA_PREFIX}/bin" -mindepth 1 -type f -name '*legion*' || true)" \
       "$(find "${CONDA_PREFIX}/bin" -mindepth 1 -type f -name 'legate' || true)" \
       "$(find "${CONDA_PREFIX}/bin" -mindepth 1 -type f -name 'bind.sh' || true)" \
       "$(find "${CONDA_PREFIX}/bin" -mindepth 1 -type f -name 'lgpatch' || true)" \
       ;
