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

## Print this help message
.PHONY: help
help: default_help

## -- Commonly used options --

## Build the library
##
## Options:
## - VERBOSE=0|1                  - whether to enable a verbose build
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: all
all: default_all

## Remove build artifacts
##
## Options:
## - VERBOSE=0|1                  - whether to enable a verbose clean
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: clean
clean: default_clean
	@$(CMAKE) -E rm -rf -- $(LEGATE_CORE_DIR)/legate_core.egg-info

## Install the library
##
## Options:
## - VERBOSE=0|1                  - whether to enable a verbose install
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: install
install: default_install

## Create an installable package of the library
##
## Options:
## - VERBOSE=0|1                  - whether to enable a verbose build
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: package
package: default_package

## Run clang-tidy over the repository
##
## Options:
## - VERBOSE=0|1                  - whether to enable a verbose build
## - LEGATE_CORE_CMAKE_ARGS='...' - any additional arguments to pass to the cmake command
##
.PHONY: tidy
tidy:
	@$(LEGATE_CORE_BUILD_COMMAND) --target tidy $(LEGATE_CORE_CMAKE_ARGS)

.PHONY: cpp-docs
cpp-docs:
	@$(LEGATE_CORE_BUILD_COMMAND) --target doxygen_legate $(LEGATE_CORE_CMAKE_ARGS)

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

.PHONY: docs
docs: cpp-docs py-docs
