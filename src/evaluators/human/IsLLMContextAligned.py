# IsLLMContextAligned.py
#
# Description: Is the LLM comment contextually aligned with the diff?
#              Checks that the comment does not hallucinate content not present
#              in the diff (context consistency check).
#              Fetches diff from the benchmark dataset at evaluation time.

import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

COMMIT_DIFF_BASE = Path("benchmarks/contextcrbench/input-dataset/commit_diff")
MODEL = "gpt-5.1-2025-11-13"

SYSTEM_PROMPT = """\
[Task Description]
You are an LLM-Judge in software engineering tasked with directly assessing whether a review comment (the "claims") is context-misaligne with a given code diff (the "facts").

[Definition]
Context misalignment occurs when an LLM-generated code review comment contains information, claims, or suggestions that:
- Are not supported by, or cannot be traced to, the relevant context (e.g., code diff, or input text).
- Contradict the facts or intent present in the context.
- Reference information that is missing, irrelevant, or outside the scope of the provided context.

[Scoring Guide]
0) Fully aligned; all claims are directly supported and consistent with the code diff and intent.
1) Mostly aligned; most claims are supported, minor inconsistencies.
2) Partially aligned; some claims are supported, others are not.
3) Mostly misaligned; very little support from the code diff.
4) Completely misaligned; claims are unsupported or contradicted by the code diff.

[Output]
Provide your evaluation in the following valid JSON format (no extra text) using a scores-based evaluation of context misalignment levels (0-4), along with a concise explanation referencing specific evidence from the code diff and the review comment.

[Example Output]
{ "answer": 0-4, "explanation": "..."}
"""

USER_PROMPT_TEMPLATE = """\
[Diff]{commit_diff}[/Diff]
[LLM Code Review]{llm_comment}[/LLM Code Review]

Please evaluate whether this review comment is context-aligned or context-misaligned with the given code diff. Only provide the valid JSON format as specified in the system prompt.
"""


def _load_commit_diff(diff_id: str) -> str | None:
    """Load commit diff from centralized commit_diff directory."""
    commit_diff_path = COMMIT_DIFF_BASE / f"{diff_id}.diff"
    if commit_diff_path.exists():
        return commit_diff_path.read_text()

    return None


def _check_context_aligned(commit_diff: str, llm_comment: str) -> bool | None:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        commit_diff=commit_diff,
        llm_comment=llm_comment,
    )
    print(f"[IsLLMContextAligned]")
    print(f"  LLM Comment : {llm_comment!r}")
    print(f"  Diff length : {len(commit_diff)} chars")

    import json

    for attempt in range(1, 4):
        result = single_request(
            model="gpt-5.1-2025-11-13",
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

    print(f"  Result      : None (all 3 attempts failed)")
    return None


class IsLLMContextAligned(BaseEvaluator):
    evaluation_name = "metric/human/is_llm_context_aligned"

    @staticmethod
    def evaluate(llm_comment_dict: dict, ground_truth_dict: list[dict]) -> bool | None:

        diff_id = llm_comment_dict.get("diff_id", "")
        llm_comment = llm_comment_dict.get("comment", "")

        if not diff_id or not llm_comment:
            return None

        commit_diff = _load_commit_diff(diff_id)
        if commit_diff is None:
            print(
                f"[IsLLMContextAligned] commit.diff not found for diff_id={diff_id!r}"
            )
            return None

        try:
            return _check_context_aligned(
                commit_diff=commit_diff, llm_comment=llm_comment
            )
        except Exception as ex:
            print(f"[IsLLMContextAligned] Error: {ex}")
            return None
