# iterate hunks given diff string

import re
import json
from pathlib import Path

import whatthepatch


def iterate_hunks(diff_str: str) -> tuple[list, list]:
    """
    returns individual hunks at file and block levels from raw patch
    only includes added/modified files, ignores deleted files
    """

    file_hunks = []
    block_hunks = []

    try:

        for diff in whatthepatch.parse_patch(diff_str):

            # Skip deleted files (check multiple indicators for robustness)
            is_deleted = (
                diff.header.new_path is None
                or diff.header.new_path == "/dev/null"
                or diff.header.new_path == "dev/null"
                or "deleted file mode" in diff.text
            )
            if is_deleted:
                continue

            changes = diff.changes
            hunk_ind = 1  # hunk index starts at 1
            block_hunk_lines = []

            # file hunk from original diff
            file_hunks.append(diff.text)
            old_file_path = f"--- a/{diff.header.old_path}\n"
            new_file_path = f"+++ b/{diff.header.new_path}\n"

            # try keeping hunk headers
            # only support unified diff (git's default)
            match_block_hunk_headers = re.findall(
                pattern=r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)$",
                string=diff.text,
                flags=re.MULTILINE,
            )

            # assemble block hunk headers
            block_hunk_headers = [
                f"@@ -{m[0]},{m[1]} +{m[2]},{m[3]} @@{m[4]}$"
                for m in match_block_hunk_headers
            ]

            # block hunk, breaks file_hunk
            for i, change in enumerate(changes):
                current_hunk_header = (
                    f"{block_hunk_headers[hunk_ind - 1]}\n"
                    if hunk_ind - 1 < len(block_hunk_headers)
                    else ""
                )

                if hunk_ind != change.hunk:
                    # append collected hunk headers
                    block_hunks.append(
                        old_file_path
                        + new_file_path
                        + current_hunk_header
                        + "\n".join(block_hunk_lines)
                    )
                    hunk_ind = change.hunk
                    block_hunk_lines = []

                if change.old != None and change.new != None:
                    # unchanged line, keep leading space
                    block_hunk_lines.append(f" {change.line}")
                elif change.new == None:
                    # deletion / modified
                    block_hunk_lines.append(f"-{change.line}")
                elif change.old == None:
                    # addition / modified
                    block_hunk_lines.append(f"+{change.line}")

                # last hunk
                if i == len(changes) - 1:
                    block_hunks.append(
                        old_file_path
                        + new_file_path
                        + current_hunk_header
                        + "\n".join(block_hunk_lines)
                    )

    except Exception as e:
        print("Error occured:", str(e))

    return file_hunks, block_hunks


def test_iterate_hunks():
    """
    unit test iterate_hunks() with test_diff.json
    """
    with open(Path(__file__).with_name("test_diff.json"), "r") as file:
        data = json.load(file)
        # Updated expected counts: [file_hunks, block_hunks]
        # Test case 0: Original test (2 modified files, 4 block hunks)
        # Test case 1: Original test (1 modified file, 2 block hunks)
        # Test case 2: Mixed operations (1 modified + 1 new file, 2 deleted files filtered out)
        hunk_length = [[2, 4], [1, 2], [2, 2], [6, 6]]

        for i, diff in enumerate(data["diff"]):
            file_hunks, block_hunks = iterate_hunks(diff_str=diff)

            print(f"\n========== TEST CASE {i} ==========")
            print(
                f"Expected: {hunk_length[i][0]} file hunks, {hunk_length[i][1]} block hunks"
            )
            print(
                f"Actual: {len(file_hunks)} file hunks, {len(block_hunks)} block hunks"
            )

            assert (
                len(file_hunks) == hunk_length[i][0]
            ), f"Test case {i}: Expected {hunk_length[i][0]} file hunks, got {len(file_hunks)}"
            assert (
                len(block_hunks) == hunk_length[i][1]
            ), f"Test case {i}: Expected {hunk_length[i][1]} block hunks, got {len(block_hunks)}"

            # Show which files were included (for verification)
            if i == 2:  # The new test case with deleted files
                print("Files included in test case 2 (should exclude deleted files):")
                for j, hunk in enumerate(file_hunks):
                    lines = hunk.split("\n")
                    for line in lines:
                        if line.startswith("diff --git"):
                            print(f"  File {j+1}: {line}")
                            break

            # usage example
            """
            print("========== file hunks")
            for hunk in file_hunks:
                print(hunk)
                print("end_hunk ===============\n\n")
            print("========== block hunks")
            for hunk in block_hunks:
                print(hunk)
                print("end_hunk===============\n\n")
            print("========================================")
            # """

            print(file_hunks)
            print(block_hunks)


def annotate_hunk_line_number(hunk: str) -> str:
    """
    prepends line number on each line of an individual hunk
    TODO integrate into iterate hunks if neceassary
    assumptions:
        - hunk is parsed successfully by this util
        - ignore deleted lines
    """

    CHUNK_FILE_SEP = ["---", "+++"]
    DIFF_START = "diff --git"

    hunk_lines = hunk.split("\n")
    old_line = -1
    new_line = -1
    for i, line in enumerate(hunk_lines):

        # new hunk
        new_hunk = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)$", line)
        if new_hunk:
            old_line = int(new_hunk.groups()[0])
            new_line = int(new_hunk.groups()[2])

        elif any(line.strip().startswith(c) for c in CHUNK_FILE_SEP) or line.startswith(
            DIFF_START
        ):
            # end of hunk
            new_line = -1
            old_line = -1

        else:
            # appending line number
            if line.startswith("+") or line.startswith(" "):
                # new / modified line appearing in new version
                prepending = str(new_line).rjust(12)
                hunk_lines[i] = "".join([line[:1], prepending, ":", line[1:]])

            elif line.startswith("-"):
                # removed line
                prepending = str(old_line).ljust(12)
                hunk_lines[i] = "".join([line[:1], " ", prepending, ":", line[1:]])

            # manage line counter
            new_line += 1 if line.startswith("+") or line.startswith(" ") else 0
            old_line += 1 if line.startswith("-") or line.startswith(" ") else 0

    return "\n".join(hunk_lines)


def test_annotate_hunk_line_number():
    """
    sanity test annotate_hunk_line_number() with test_diff.json
    """
    with open(Path(__file__).with_name("test_diff.json"), "r") as file:
        data = json.load(file)

        for diff in data["diff"]:
            file_hunks, block_hunks = iterate_hunks(diff_str=diff)

            for hunk in file_hunks:
                annotated_hunk = annotate_hunk_line_number(hunk=hunk)
                print(annotated_hunk)
                print("===========")
            print("-------------------")
            for hunk in block_hunks:
                annotated_hunk = annotate_hunk_line_number(hunk=hunk)
                print(annotated_hunk)
                print("===========")


def mask_filename_in_patch(hunk: str) -> str:
    """
    replace file path in patch diff (for benchmark dataset)
    """

    CHUNK_FILE_SEP = ["---", "+++"]

    hunk_lines = hunk.split("\n")
    for i, line in enumerate(hunk_lines):
        if any(line.strip().startswith(c) for c in CHUNK_FILE_SEP):
            # --- a/a05_security_misconfiguration/javascript/Xxe_cwe611.js
            # +++ b/a05_security_misconfiguration/javascript/Xxe_cwe611.js
            line_elem = line.split(" ")
            path_initial = line_elem[1].split("/")[0]
            file_extention = (
                f".{line_elem[1].split(".")[-1]}"
                if len(line_elem[1].split(".")) > 1
                else ""
            )
            hunk_lines[i] = " ".join(
                [line_elem[0], f"{path_initial}/dummy_file_path{file_extention}"]
            )

    return "\n".join(hunk_lines)


def test_mask_filename_in_patch():
    """
    sanity test mask_filename_in_patch() with test_diff.json
    """
    with open(Path(__file__).with_name("test_diff.json"), "r") as file:
        data = json.load(file)

        for diff in data["diff"]:
            file_hunks, block_hunks = iterate_hunks(diff_str=diff)

            for hunk in file_hunks:
                masked_hunk = mask_filename_in_patch(hunk=hunk)
                print(masked_hunk)
                print("===========")
            print("-------------------")
            for hunk in block_hunks:
                annotated_hunk = mask_filename_in_patch(hunk=hunk)
                print(annotated_hunk)
                print("===========")


if __name__ == "__main__":
    # unit test
    test_iterate_hunks()
    test_annotate_hunk_line_number()
    test_mask_filename_in_patch()
