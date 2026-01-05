# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# begin_group: Start a named section of log output, possibly with color.
# Usage: begin_group "Group Name" [Color]
#   Group Name: A string specifying the name of the group.
#   Color (optional): ANSI color code to set text color. Default is blue (1;34).
function begin_group()
{
  # See options for colors here: https://gist.github.com/JBlond/2fea43a3049b38287e5e9cefc87b2124
  local blue='34'
  local name="${1:-'(unnamed group)'}"
  local color="${2:-${blue}}"

  if [[ "${LEGATE_CI_GROUP:-}" == '' ]]; then
    LEGATE_CI_GROUP=0
  fi

  if [[ "${LEGATE_CI_GROUP}" == '0' ]]; then
    echo -e "::group::\e[${color}m${name}\e[0m"
  else
    echo -e "\e[${color}m== ${name} ===========================================================================\e[0m"
  fi
  export LEGATE_CI_GROUP=$((LEGATE_CI_GROUP+1))
}
export -f begin_group

# end_group: End a named section of log output and print status based on exit status.
# Usage: end_group "Group Name" [Exit Status]
#   Group Name: A string specifying the name of the group.
#   Exit Status (optional): The exit status of the command run within the group. Default is 0.
function end_group()
{
  local name="${1:-'(unnamed group)'}"
  local build_status="${2:-0}"
  local duration="${3:-}"
  local red='31'
  local blue='34'

  if [[ "${LEGATE_CI_GROUP:-}" == '' ]]; then
    echo 'end_group called without matching begin_group!'
    exit 1
  fi

  export LEGATE_CI_GROUP=$((LEGATE_CI_GROUP-1))
  if [[ "${LEGATE_CI_GROUP}" == '0' ]]; then
    echo -e "::endgroup::\e[${blue}m (took ${duration})\e[0m"
  else
    echo -e "\e[${blue}m== ${name} ===========================================================================\e[0m"
  fi

  if [[ "${build_status}" != '0' ]]; then
    local fail_msg="Failed (⬆️ click above for full log ⬆️)"  # legate-lint: no-ascii-only
    echo -e "::error::\e[${red}m ${name} - ${fail_msg}\e[0m"
    exit "${build_status}"
  fi
}
export -f end_group

# Runs a command within a named group, handles the exit status, and prints appropriate
# messages based on the result.
# Usage: run_command "Group Name" command [arguments...]
function run_command()
{
  { set +x; } 2>/dev/null;
  local old_opts
  old_opts=$(set +o)
  set +e

  local group_name="${1:-}"
  shift
  local command=("$@")
  local status

  begin_group "${group_name}"
  local start_time
  start_time=$(date +%s)
  rapids-logger "Running command: " "${command[@]}"
  "${command[@]}"
  status=$?
  # In case the command enables either of these, we want to disable them so that we can
  # finish up here -- we will be restoring the old options at function end anyways.
  { set +xe; } 2>/dev/null;
  local end_time
  end_time=$(date +%s)
  local duration
  duration=$((end_time - start_time))
  end_group "${group_name}" "${status}" "${duration}"
  eval "${old_opts}"
  return "${status}"
}
export -f run_command
