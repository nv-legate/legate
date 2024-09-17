#!/usr/bin/env python3
# dej2lint - remove jinja2 directives for lint checking
# also should try to fix line nums in lint output
# currently only runs on yaml files with yamllint

import argparse
import re
import sys

import yaml
from yamllint import linter
from yamllint.config import YamlLintConfig


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process YAML files with yamllint, removing Jinja2 "
        "directives.",
        epilog='Example: %(prog)s --config-data "{extends: default, '
        'rules: {line-length: disable}}" file.yaml',
    )
    parser.add_argument(
        "file_paths", nargs="+", help="Paths to YAML files to process."
    )
    parser.add_argument(
        "-c",
        "--config-file",
        help="Path to YAML configuration file for yamllint.",
    )
    parser.add_argument(
        "--config-data", help="Custom YAML configuration for yamllint."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output."
    )
    return parser.parse_args()


def process_file(file_path, config, debug):
    with open(file_path, "r") as infile:
        lines = infile.readlines()

    cleanbuf = []
    line_mapping = {}
    original_linenum = 0
    cleaned_linenum = 0

    for line in lines:
        original_linenum += 1

        if re.search(r"^\{#.*#\}$", line) or re.search(r"^\{%.*%\}$", line):
            continue

        cleanline = re.sub(r"\{#.*#\}", "", line)
        cleanline = re.sub(r"\{%.*%\}", "", cleanline)
        cleanline = re.sub(r"\{\{.*\}\}", "lintremove", cleanline)

        # Remove leading space before comment in Jinja lines
        if re.search(r"\{[#%]", line):
            cleanline = re.sub(r"(\s+)#", "#", cleanline)

        if cleanline.strip():
            cleanbuf.append(cleanline.rstrip())
            cleaned_linenum += 1
            line_mapping[cleaned_linenum] = original_linenum

    cleaned_content = "\n".join(cleanbuf) + "\n"

    if debug:
        print(f"Processing file: {file_path}")
        print("---")
        print(cleaned_content)
        print("---")

    # Run the linter and print with original line numbers
    issues_found = False
    for item in linter.run(cleaned_content, config, file_path):
        issues_found = True
        origline = line_mapping.get(item.line, item.line)
        print(
            f"{file_path}:{origline}:{item.column}: "
            f"{item.level}: {item.message}"
        )

    if debug:
        print("\nLine mapping:")
        for clean_line, orig_line in sorted(line_mapping.items()):
            print(f"Clean: {clean_line} -> Original: {orig_line}")

    return issues_found


def main():
    if len(sys.argv) == 1:
        print(
            "Error: No arguments provided. Use -h or --help for usage "
            "information."
        )
        sys.exit(1)

    args = parse_arguments()

    # Use configuration file if provided
    if args.config_file:
        conf = YamlLintConfig(file=args.config_file)
    # Use custom configuration data if provided
    elif args.config_data:
        config_dict = yaml.safe_load(args.config_data)
        conf = YamlLintConfig(content=yaml.dump(config_dict))
    # Use default configuration
    else:
        conf = YamlLintConfig("extends: default")

    issues_found = False
    for file_path in args.file_paths:
        if process_file(file_path, conf, args.debug):
            issues_found = True

    if issues_found:
        sys.exit(1)


if __name__ == "__main__":
    main()
