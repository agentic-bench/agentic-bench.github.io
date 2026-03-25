"""
IsLLMHunkContextAligned Evaluator

This evaluator assesses whether LLM-generated comments are contextually aligned
with the SPECIFIC HUNK they reference in the commit diff (not the entire diff).

This is a focused version of IsLLMContextAligned that extracts only the relevant
hunk containing the commented line, reducing noise from unrelated changes.
"""

import os
import sys
import json
from pathlib import Path
from typing import Any

import whatthepatch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

COMMIT_DIFF_BASE = Path("benchmarks/agenticcr-verified/input-dataset/commit_diff")
MODEL = "gpt-5.1-2025-11-13"

# System prompt for the LLM judge
SYSTEM_PROMPT = """\
[Task Description]
You are an LLM-Judge in software engineering tasked with directly assessing whether a review comment (the "claims") is context-misaligned with a given code diff hunk (the "facts").

[Definition]
Context misalignment occurs when an LLM-generated code review comment contains information, claims, or suggestions that:
- Are not supported by, or cannot be traced to, the relevant context (the diff hunk).
- Contradict the facts or intent present in the diff hunk.
- Reference information that is missing, irrelevant, or outside the scope of the provided hunk.

[Scoring Guide]
0) Fully aligned; all claims are directly supported and consistent with the code diff hunk.
1) Mostly aligned; most claims are supported, minor inconsistencies.
2) Partially aligned; some claims are supported, others are not.
3) Mostly misaligned; very little support from the code diff hunk.
4) Completely misaligned; claims are unsupported or contradicted by the code diff hunk.

[Output]
Provide your evaluation in the following valid JSON format (no extra text) using a scores-based evaluation of context misalignment levels (0-4), along with a concise explanation referencing specific evidence from the code diff hunk and the review comment.

[Example Output]
{ "answer": 0-4, "explanation": "..."}
"""

# User prompt template
USER_PROMPT_TEMPLATE = """\
[Diff Hunk]
{diff_hunk}
[/Diff Hunk]

[LLM Code Review]
{llm_comment}
[/LLM Code Review]

Please evaluate whether this review comment is context-aligned or context-misaligned with the given code diff hunk. Only provide the valid JSON format as specified in the system prompt.
"""


def _load_commit_diff(diff_id: str) -> str | None:
    """
    Load commit diff from centralized commit_diff directory.

    Args:
        diff_id: ID of the diff (e.g., "repo_issue_123_pr_456")

    Returns:
        Diff content as string, or None if file not found
    """
    try:
        commit_diff_path = COMMIT_DIFF_BASE / f"{diff_id}.diff"
        if commit_diff_path.exists():
            return commit_diff_path.read_text(encoding="utf-8")
        return None
    except Exception as e:
        print(f"[IsLLMHunkContextAligned] Error loading diff {diff_id}: {e}")
        return None


def _extract_target_hunk(
    commit_diff: str, comment_file: str, comment_line: int
) -> str | None:
    """
    Extract the specific hunk from the commit diff that contains the commented line.

    Args:
        commit_diff: Full commit diff content
        comment_file: File path that was commented on (e.g., "src/components/LibraryUnit.tsx")
        comment_line: Line number that was commented on

    Returns:
        String containing just the target hunk in unified diff format, or None if not found
    """
    try:
        patches = list(whatthepatch.parse_patch(commit_diff))

        for patch in patches:
            # Match the file path
            new_path = patch.header.new_path if patch.header else None
            if not new_path:
                continue

            # Remove leading 'b/' from git diff format
            if new_path.startswith("b/"):
                new_path = new_path[2:]

            # Check if this is the file we're looking for
            if new_path != comment_file:
                continue

            # Group changes by hunk number
            hunks = {}
            for change in patch.changes:
                hunk_num = change.hunk
                if hunk_num not in hunks:
                    hunks[hunk_num] = []
                hunks[hunk_num].append(change)

            # Search through each hunk for the target line
            for hunk_num, hunk_changes in hunks.items():
                # Check if this hunk contains the comment line
                contains_line = any(
                    change.new == comment_line for change in hunk_changes
                )

                if not contains_line:
                    continue

                # Build the hunk output
                hunk_lines = []

                # Add file headers
                old_path = (
                    patch.header.old_path if patch.header else f"a/{comment_file}"
                )
                new_path_full = (
                    patch.header.new_path if patch.header else f"b/{comment_file}"
                )
                hunk_lines.append(f"--- {old_path}")
                hunk_lines.append(f"+++ {new_path_full}")

                # Calculate hunk header
                old_start = min(
                    (c.old for c in hunk_changes if c.old is not None), default=0
                )
                new_start = min(
                    (c.new for c in hunk_changes if c.new is not None), default=0
                )
                old_count = sum(1 for c in hunk_changes if c.old is not None)
                new_count = sum(1 for c in hunk_changes if c.new is not None)

                hunk_lines.append(
                    f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"
                )

                # Add the changes
                for change in hunk_changes:
                    hunk_lines.append(change.line)

                return "\n".join(hunk_lines)

        return None

    except Exception as e:
        print(
            f"[IsLLMHunkContextAligned] Error extracting hunk for {comment_file}:{comment_line}: {e}"
        )
        return None


def _check_context_aligned(diff_hunk: str, llm_comment: str) -> bool | None:
    """
    Use LLM judge to check if the comment is contextually aligned with the diff hunk.

    Args:
        diff_hunk: The specific diff hunk in unified diff format
        llm_comment: The review comment text

    Returns:
        True if aligned (score 0), False if not aligned, None if LLM call fails
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        diff_hunk=diff_hunk, llm_comment=llm_comment
    )

    print(f"[IsLLMHunkContextAligned]")
    print(f"  LLM Comment : {llm_comment!r}")
    print(f"  Hunk length : {len(diff_hunk)} chars")

    # Try up to 3 times
    for attempt in range(1, 4):
        try:
            result = single_request(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_output=True,
            )

            if result is None:
                print(f"  Attempt {attempt}: None (API error)")
                continue

            print(f"  Attempt {attempt} raw output: {result!r}")

            try:
                parsed = json.loads(result) if isinstance(result, str) else result
                score = int(parsed["answer"])
                aligned = score == 0
                print(f"  Score       : {score} -> aligned={aligned}")
                return aligned
            except Exception as ex:
                print(f"  Attempt {attempt} parse error: {ex}")
                continue

        except Exception as e:
            print(f"  Attempt {attempt} error: {e}")
            continue

    print(f"  Result      : None (all 3 attempts failed)")
    return None


class IsLLMHunkContextAligned(BaseEvaluator):
    """
    Evaluator that checks if LLM comments are contextually aligned with
    the specific hunk they reference in the commit diff.

    Returns:
        bool: True if contextually aligned, False if not
        None: If hunk cannot be extracted or evaluation fails
    """

    evaluation_name = "metric/human/is_llm_hunk_context_aligned"

    @staticmethod
    def evaluate(llm_comment_dict: dict, ground_truth_dict: list[dict]) -> bool | None:
        """
        Evaluate if the LLM comment is contextually aligned with its diff hunk.

        Args:
            llm_comment_dict: Contains 'diff_id', 'comment_file', 'comment_line', and 'comment' keys
            ground_truth_dict: Ground truth data (not used in this evaluator)

        Returns:
            True if aligned, False if not aligned, None if cannot evaluate
        """
        try:
            # Extract required fields
            diff_id = llm_comment_dict.get("diff_id", "")
            comment_file = llm_comment_dict.get("comment_file")
            comment_line = llm_comment_dict.get("comment_line")
            comment_text = llm_comment_dict.get("comment", "")

            # Validate inputs
            if not all([diff_id, comment_file, comment_line, comment_text]):
                return None

            # Load the full commit diff
            commit_diff = _load_commit_diff(diff_id)
            if not commit_diff:
                print(
                    f"[IsLLMHunkContextAligned] commit.diff not found for diff_id={diff_id!r}"
                )
                return None

            # Extract the specific hunk containing the commented line
            diff_hunk = _extract_target_hunk(
                commit_diff, comment_file, int(comment_line)
            )

            if not diff_hunk:
                # Could not find the hunk - skip evaluation
                print(
                    f"[IsLLMHunkContextAligned] Could not extract hunk for {comment_file}:{comment_line}"
                )
                return None

            # Check context alignment using LLM judge
            is_aligned = _check_context_aligned(diff_hunk, comment_text)

            return is_aligned

        except Exception as e:
            # Log error but don't crash - return None to skip this evaluation
            print(f"[IsLLMHunkContextAligned] Error: {e}")
            import traceback

            traceback.print_exc()
            return None
