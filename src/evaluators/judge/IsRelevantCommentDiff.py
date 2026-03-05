# IsRelevantCommentDiff.py
#
# Description: Is the LLM comment relevant to the code diff?
#              Requires _diff injected into comment_dict by evaluator.py.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

MODEL = "gpt-4.1-2025-04-14"

SYSTEM_PROMPT = """\
You are an expert code reviewer evaluating the relevancy of LLM comments to code diffs.

A comment is RELEVANT if it:
- Addresses content that is being added, modified, or removed in the diff
- Suggests improvements to the new/changed code
- Points out potential issues with the changes being made

A comment is NOT RELEVANT if it:
- Discusses code not touched by the diff
- Suggests changes to unmodified parts of the file
- Is completely unrelated to the changes

Respond with exactly 'relevant' or 'not relevant'."""

USER_PROMPT_TEMPLATE = (
    "Is the following code review comment relevant to the diff?\n\n"
    "[Diff]\n{diff}\n[/Diff]\n\n"
    "[Comment]\n{comment}\n[/Comment]\n\n"
    "Respond with 'relevant' or 'not relevant'."
)


class IsRelevantCommentDiff(BaseEvaluator):
    evaluation_name = "metric/judge/is_comment_diff_relevant"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        diff = comment_dict.get("_diff", "")
        llm_comment = comment_dict.get("comment", "")

        if not llm_comment or not diff:
            return False

        result = single_request(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_TEMPLATE.format(diff=diff, comment=llm_comment),
            json_output=False,
        )
        verdict = result is not None and result.strip().lower() == "relevant"
        try:
            logger = BaseEvaluator.get_json_logger()
            logger.info(
                "IsRelevantCommentDiff",
                extra={
                    "evaluator": "is_comment_diff_relevant",
                    "diff_id": comment_dict.get("diff_id"),
                    "raw": result,
                    "result": verdict,
                },
            )
        except FileNotFoundError:
            pass
        return verdict
