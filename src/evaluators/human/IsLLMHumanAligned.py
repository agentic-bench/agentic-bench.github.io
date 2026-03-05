# IsLLMHumanAligned.py
#
# Description: Is the LLM comment semantically aligned with any human
#              ground-truth comment on the same file within ±LINE_WINDOW?
#              Uses an LLM judge for semantic comparison.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

LINE_WINDOW = 5
MODEL = "gpt-5.1-2025-11-13"

SYSTEM_PROMPT = """\
[Task Description]
You are an LLM-Judge in software engineering tasked with directly assessing whether an LLM-generated review comment is aligned with a human-written code review comment from the same file.

[Definition]
Misalignment occurs when an LLM-generated code review comment contains information, claims, or suggestions that:
- Address a completely different issue or concern from the human reference comment.
- Are too generic or unrelated to be meaningfully compared to the human reference.

[Scoring Guide]
0) Fully aligned; the AI comment addresses the exact same issue and intent as the human comment.
1) Mostly aligned; the core concern matches, with minor differences in scope or detail.
2) Partially aligned; some overlap in intent but notable differences in the issue being raised.
3) Mostly misaligned; very little overlap — the AI comment touches on a different concern.
4) Completely misaligned; the AI comment addresses an entirely different issue or contradicts the human comment.

[Output]
Provide your evaluation in the following valid JSON format (no extra text) using a scores-based evaluation of human alignment levels (0-4), along with a concise explanation referencing specific evidence from both comments.

[Example Output]
{ "answer": 0-4, "explanation": "..."}
"""

USER_PROMPT_TEMPLATE = """\
### Human Comment:
{human_comment}

### LLM Comment:
{llm_comment}

Please evaluate whether this LLM comment is human-aligned. Only provide the valid JSON format as specified in the system prompt.
"""


def _check_human_aligned(
    llm_file: str,
    llm_comment: str,
    human_file: str,
    human_comment: str,
) -> bool | None:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        human_file=human_file,
        human_comment=human_comment,
        llm_file=llm_file,
        llm_comment=llm_comment,
    )
    print(f"[IsLLMHumanAligned]")
    print(f"  LLM  ({llm_file})  : {llm_comment!r}")
    print(f"  Human ({human_file}) : {human_comment!r}")

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

            # accept mostly aligned and above - this should prevent surface match
            aligned = score <= 1

            print(f"  Score       : {score} -> aligned={aligned}")
            return aligned

        except Exception as ex:
            print(f"  Attempt {attempt} parse error: {ex}")

    print(f"  Result      : None (all 3 attempts failed)")
    return None


class IsLLMHumanAligned(BaseEvaluator):
    evaluation_name = "metric/human/is_llm_human_aligned"

    @staticmethod
    def evaluate(llm_comment_dict: dict, ground_truth_dict: list[dict]) -> bool | None:
        if not ground_truth_dict:
            return None

        llm_comment = llm_comment_dict.get("comment", "")
        llm_file = llm_comment_dict.get("comment_file", "")

        if not llm_comment or not llm_file:
            return None

        # Only consider ground truth comments on the same file
        aligned_gts = [
            gt for gt in ground_truth_dict if llm_file == gt.get("comment_file")
        ]

        if not aligned_gts:
            return None

        try:
            results = [
                _check_human_aligned(
                    llm_file=llm_file,
                    llm_comment=llm_comment,
                    human_file=gt.get("comment_file", ""),
                    human_comment=gt.get("comment_content", ""),
                )
                for gt in aligned_gts
            ]
            # True if any aligned GT is a match
            valid = [r for r in results if r is not None]
            if not valid:
                return None
            return any(valid)

        except Exception as ex:
            print(f"[IsLLMHumanAligned] Error: {ex}")
            return None
