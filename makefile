# -*- mode: makefile-gmake -*-
ifndef LEGATE_CORE_DIR
export LEGATE_CORE_DIR := $(shell ./scripts/get_legate_core_dir.py)
endif

ifndef LEGATE_CORE_ARCH
$(error LEGATE_CORE_ARCH not defined)
endif

.SUFFIXES:
.DELETE_ON_ERROR:

.DEFAULT_GOAL := all

include $(LEGATE_CORE_DIR)/$(LEGATE_CORE_ARCH)/gmakevariables

ifeq ($(strip $(V)),1)
export VERBOSE ?= 1
else ifeq ($(strip $(V)),0)
export VERBOSE ?= 0
endif

## Common options for all commands:
##
## - VERBOSE=0|1                    - whether to enable verbose output
## - V=0|1                          - alias for VERBOSE
## - LEGATE_CORE_DIR=/absolute/path - override (or set) the root directory for Legate.Core
## - LEGATE_CORE_ARCH=directory     - override (or set) the current arch directory

## Print this help message
##
.PHONY: help
help: default_help

## Build the library
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: all
all: default_all

## Remove build artifacts
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: clean
clean: default_clean
	@$(CMAKE) -E rm -rf -- $(LEGATE_CORE_DIR)/legate_core.egg-info
	@$(CMAKE) -E rm -rf -- $(LEGATE_CORE_DIR)/$(LEGATE_CORE_ARCH)/_skbuild


.PHONY: install_private
install_private: all
	@$(MAKE) --no-print-directory default_install

## Install the library
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: install
install: install_private

.PHONY: package_private
package_private: all
	@$(MAKE) --no-print-directory default_package

## Create an installable package of the library
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: package
package: package_private

## Run clang-tidy over the repository
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: tidy
tidy:
	@$(LEGATE_CORE_BUILD_COMMAND) --target tidy $(LEGATE_CORE_CMAKE_ARGS)

## Run clang-tidy only over the files which have been changed by your branch.
##
## Beware that this target may not fully work. Turns out asking git "which base branch is
## my current branch based off of" is more or less impossible to answer reliably... so
## depending on which branch it selects, this might not do what you want. It may try to
## check more files than needed (worst case, all of them), or it may not check all
## files. Buyer beware.
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: tidy-diff
tidy-diff:
	@$(LEGATE_CORE_BUILD_COMMAND) --target tidy-diff $(LEGATE_CORE_CMAKE_ARGS)

## Generate raw doxygen output
##
.PHONY: doxygen
doxygen:
	@$(LEGATE_CORE_BUILD_COMMAND) --target Doxygen $(LEGATE_CORE_CMAKE_ARGS)

## Build combined Sphinx documentation
##
.PHONY: docs
docs: doxygen
	@$(LEGATE_CORE_BUILD_COMMAND) --target Sphinx $(LEGATE_CORE_CMAKE_ARGS)
