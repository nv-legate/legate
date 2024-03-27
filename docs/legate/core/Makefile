# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?= -v -W
PARALLEL_BUILD ?= 1
SPHINXBUILD    ?= $(shell which sphinx-build)
SOURCEDIR      = source
BUILDDIR       = build

.PHONY: help clean html linkcheck

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  all              to clean build standalone HTML files and run linkcheck"
	@echo "  clean            to clear all built documentation files"
	@echo "  html             to make standalone (non-gallery) HTML files"
	@echo "  linkcheck        to run the link checker on built docs"

all: clean html linkcheck

clean:
	rm -rf build/html/docs
	find source/ -name "generated" | xargs rm -rf

html:
	@start=$$(date +%s) \
	; $(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) -j $(PARALLEL_BUILD) \
	&& mkdir -p build/html/docs \
	&& echo "\nBuild finished in $$(($$(date +%s)-start)) seconds\n"

linkcheck: html
	@start=$$(date +%s) \
	; $(SPHINXBUILD) "$(SOURCEDIR)" "$(BUILDDIR)" -b linkcheck \
	; echo "\nLink check finished in $$(($$(date +%s)-start)) seconds. Any broken links are listed below:\n" \
	&& [ -f "$(BUILDDIR)/output.txt" ] && cat "$(BUILDDIR)/output.txt"
