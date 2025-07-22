#!/usr/bin/env python3
"""
Selective Test Filter Tool

Given two git refs (branches or commit IDs), this tool:
- Identifies changed files between the refs
- Loads the enhanced dependency mapping JSON (from enhanced_ninja_parser.py)
- Maps changed files to affected test executables (optionally filtering for "test_" prefix)
- Exports the list of tests to run to tests_to_run.json

Usage:
  python selective_test_filter.py <depmap_json> <ref1> <ref2> [--all | --test-prefix] [--output <output_json>]

Arguments:
  <depmap_json>   Path to enhanced_dependency_mapping.json
  <ref1>          Source git ref (branch or commit)
  <ref2>          Target git ref (branch or commit)

Options:
  --all           Include all executables (default)
  --test-prefix   Only include executables starting with "test_"
  --output        Output JSON file (default: tests_to_run.json)
"""

import sys
import subprocess
import json
import os
import argparse


def parser_add_arguments(parser):
    parser.add_argument("depmap_json", help="Path to dependency mapping JSON")
    parser.add_argument("ref1", help="Source git ref (branch or commit)")
    parser.add_argument("ref2", help="Target git ref (branch or commit)")
    parser.add_argument("--all", action="store_true", help="Include all executables")
    parser.add_argument(
        "--test-prefix",
        action="store_true",
        help="Only include executables starting with 'test_'",
    )
    parser.add_argument(
        "--output",
        dest="output_json",
        help="Output JSON file",
        default="tests_to_run.json",
    )


def get_changed_files(ref1, ref2):
    """Return a set of files changed between two git refs."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", ref1, ref2],
            capture_output=True,
            text=True,
            check=True,
        )
        files = set(line.strip() for line in result.stdout.splitlines() if line.strip())
        return files
    except subprocess.CalledProcessError as e:
        print(f"Error running git diff: {e}")
        sys.exit(1)


def load_depmap(depmap_json):
    """Load the dependency mapping JSON."""
    with open(depmap_json, "r") as f:
        data = json.load(f)
    # Support both old and new formats
    if "file_to_executables" in data:
        return data["file_to_executables"]
    return data


def select_exec(file_to_executables, changed_files, filter_mode):
    """Return a set of test executables affected by changed files."""
    affected = set()
    for f in changed_files:
        if f in file_to_executables:
            for exe in file_to_executables[f]:
                if filter_mode == "all":
                    affected.add(exe)
                elif filter_mode == "test_prefix" and exe.startswith("test_"):
                    affected.add(exe)
    return sorted(affected)


def main(
    depmap_json,
    ref1,
    ref2,
    all=False,
    test_prefix=False,
    output_json="tests_to_run.json",
):

    filter_mode = "all"
    if test_prefix:
        filter_mode = "test_prefix"
    if all:
        filter_mode = "all"

    if not os.path.exists(depmap_json):
        print(f"Dependency map JSON not found: {depmap_json}")
        sys.exit(1)

    changed_files = get_changed_files(ref1, ref2)
    changed_files = ['include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_lds_pipeline.hpp']
    if not changed_files:
        print("No changed files detected.")
        tests = []
    else:
        file_to_executables = load_depmap(depmap_json)
        tests = select_exec(file_to_executables, changed_files, filter_mode)
    
    import pdb
    pdb.set_trace()

    with open(output_json, "w") as f:
        json.dump(
            {"tests_to_run": tests, "changed_files": sorted(changed_files)}, f, indent=2
        )

    print(f"Exported {len(tests)} tests to run to {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser_add_arguments(parser)
    args = parser.parse_args()
    main(**vars(args))
