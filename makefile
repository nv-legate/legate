# -*- mode: makefile-gmake -*-
ifndef LEGATE_CORE_DIR
$(error LEGATE_CORE_DIR not defined!)
endif

ifndef LEGATE_CORE_ARCH
$(error LEGATE_CORE_ARCH not defined!)
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

## Install the library
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: install
install: default_install

## Create an installable package of the library
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: package
package: default_package

## Run clang-tidy over the repository
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: tidy
tidy:
	@$(LEGATE_CORE_BUILD_COMMAND) --target tidy $(LEGATE_CORE_CMAKE_ARGS)

## Build only the C++ documentation
##
## Options:
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: cpp-docs
cpp-docs:
	@$(LEGATE_CORE_BUILD_COMMAND) --target doxygen_legate $(LEGATE_CORE_CMAKE_ARGS)

## Build only the Python binding documentation
##
.PHONY: py-docs
py-docs:
ifeq ($(LEGATE_CORE_USE_PYTHON),1)
	@ret=`$(PYTHON) \
         -c \
         "import sys; sys.path.pop(0); import pkgutil; print(1 if pkgutil.get_loader('legate') else 0)"`; \
    if [ "$${ret}"  = "0" ]; then \
      echo "--- ERROR: Must install python bindings before building Python documentation!"; \
      exit 1; \
    fi
	@$(MAKE) -C $(LEGATE_CORE_DIR)/docs/legate/core html
	@$(MAKE) -C $(LEGATE_CORE_DIR)/docs/legate/core linkcheck
else
	@echo "$(LEGATE_CORE_ARCH) not configured for python, skipping docs build"
endif

## Build all available documentation
##
.PHONY: docs
docs: cpp-docs py-docs
