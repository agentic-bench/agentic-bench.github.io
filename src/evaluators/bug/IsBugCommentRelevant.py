# IsBugCommentRelevant.py
#
# Description: Is the LLM comment relevant to the target bug (CWE)?
#              Uses an LLM judge to compare the comment against ground-truth
#              CWE metadata and bug description.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

MODEL = "gpt-4.1-2025-04-14"

SYSTEM_PROMPT = (
    "You are a highly skilled software engineer.\n"
    " Your task is to determine the relevancy of given code review comment against "
    "Common Weakness Enumeration (CWE). If any information is missing, only rely on "
    "available information.\n\n"
    "You must respond with either `True`, when comment is relevant, or `False`, when "
    "it is not, without any explanation."
)

USER_PROMPT_TEMPLATE = (
    "Is code review comment below relevant to the given ground-truth CWE-ID, CWE name, "
    "CWE description, and Bug description?\n\n"
    "[Code Review Comment]\n{comment}\n\n"
    "[CWE-ID]\n{cwe_id}\n\n"
    "[CWE Name]\n{cwe_name}\n\n"
    "[CWE Description]\n{cwe_description}\n\n"
    "[Bug Description]\n{bug_description}\n\n"
    "Only respond with either `True`, when comment is relevant, or `False`, when it is "
    "not, without any explanation."
)


class IsBugCommentRelevant(BaseEvaluator):
    evaluation_name = "metric/bug/is_bug_comment_relevant"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        llm_comment = comment_dict.get("comment")
        if not llm_comment or not ground_truth:
            return False

        try:
            logger = BaseEvaluator.get_json_logger()
        except FileNotFoundError:
            logger = None

        results = []
        for gt in ground_truth:
            user_prompt = USER_PROMPT_TEMPLATE.format(
                comment=llm_comment,
                cwe_id=gt.get("cwe_id", "n/a"),
                cwe_name=gt.get("cwe_name", "n/a"),
                cwe_description=str(gt.get("cwe_description", "n/a")).strip() or "n/a",
                bug_description=gt.get("bug_description", "n/a"),
            )
            result = single_request(
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                json_output=False,
            )
            if logger:
                logger.info("IsBugCommentRelevant", extra={
                    "evaluator": "is_bug_comment_relevant",
                    "diff_id": comment_dict.get("diff_id"),
                    "cwe_id": gt.get("cwe_id"),
                    "result": result,
                })
            results.append(result)

        return "True" in results
