# -*- mode: makefile-gmake -*-
.NOTPARALLEL:

JOBS =
ifeq (4.2,$(firstword $(sort $(MAKE_VERSION) 4.2)))
  # Since make 4.2:
  #
  # * The amount of parallelism can be determined by querying MAKEFLAGS, even when
  # the job server is enabled (previously MAKEFLAGS would always contain only
  # "-j", with no number, when job server was enabled).
  #
  # https://lists.gnu.org/archive/html/info-gnu/2016-05/msg00013.html
  JOBS = $(patsubst -j%,%,$(filter -j%,$(MAKEFLAGS)))
endif

ifeq ($(strip $(JOBS)),)
  # Parse the number of jobs from inspecting process list
  MAKE_PID = $(shell echo $$PPID)
  ifeq ($(strip $(MAKE_PID)),)
    JOBS =
  else
    SED ?= sed
    PS ?= ps
    JOBS = $(strip $(shell $(PS) T | $(SED) -n 's%.*$(MAKE_PID).*$(MAKE).* \(-j\|--jobs\) *\([0-9][0-9]*\).*%\2%p'))
  endif
endif
export CMAKE_BUILD_PARALLEL_LEVEL ?= $(JOBS)

# Must go after the CMAKE_BUILD_PARALLEL_LEVEL variable above
include $(LEGATE_DIR)/$(LEGATE_ARCH)/variables.mk

export SHELL ?= /usr/bin/env bash
export AWK ?= awk

export LEGATE_BUILD_COMMAND ?= $(CMAKE) --build $(LEGATE_DIR)/$(LEGATE_ARCH)/cmake_build
export LEGATE_INSTALL_COMMAND ?= $(CMAKE) --install $(LEGATE_DIR)/$(LEGATE_ARCH)/cmake_build

ifeq ($(strip $(PREFIX)),)
export LEGATE_INSTALL_PREFIX_COMMAND = # nothing
else
export LEGATE_INSTALL_PREFIX_COMMAND = --prefix $(PREFIX)
endif

ifndef NINJA_STATUS
LEGATE_ARCH_COLOR = $(shell aedifix-select-arch-color $(LEGATE_ARCH))
COLOR_ARCH = $(shell $(CMAKE) -E cmake_echo_color --switch=$(COLOR) --$(LEGATE_ARCH_COLOR) $(LEGATE_ARCH))
export NINJA_STATUS = [%f/%t] $(COLOR_ARCH): $(SOME_UNDEFINED_VARIABLE_TO_ADD_A_SPACE)
endif

ifeq ($(strip $(V)),1)
export VERBOSE ?= 1
else ifeq ($(strip $(V)),0)
export VERBOSE ?= 0
endif
