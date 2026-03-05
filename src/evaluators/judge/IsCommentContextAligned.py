# IsCommentContextAligned.py
#
# Description: LLM judge — does the comment violate Context Inconsistency?
#              Inverts the result (True = context-consistent).
#              Requires _diff to be injected into comment_dict by evaluator.py.

import json
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

MODEL = "gpt-4.1-2025-04-14"

SYSTEM_PROMPT = """\
You are an LLM-Judge that is tasked with evaluating whether the following LLM generated code review has violated Context Inconsistency, a type of LLM hallucination.
A code review is supposed to identify issues in the code diff and potentially provide suggestions to resolve these issues.
Context Inconsistency is where the LLM generated code review hallucinated issues or suggestions that conflict with (or is not supported by) what has previously been implemented in the code diff.
Consider the possibility of incomplete context.
Always answer with True or False, before providing an explanation in the following format and make sure the JSON is valid format:
{"True_False": "", "Explanation": ""}"""


def _clean_load_judge_json(output: str) -> dict | None:
    output = output.strip()
    output = re.sub(r'\\([^"\\/bfnrtu])', r"\\\\\1", output).replace("\\", "")
    output = (
        output.replace("`", "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    )
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1:
        return None
    output = output[start : end + 1]
    try:
        return json.loads(output)
    except Exception:
        return None


class IsCommentContextAligned(BaseEvaluator):
    evaluation_name = "metric/judge/is_comment_context_aligned"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        diff = comment_dict.get("_diff", "")
        llm_comment = comment_dict.get("comment", "")
        llm_file = comment_dict.get("comment_file", "")
        llm_line = comment_dict.get("comment_line", "")

        if not llm_comment or not diff:
            return False

        user_prompt = (
            f"### After reviewing the following code diff that was previously implemented by a developer, "
            f"the LLM generated a code review at line {llm_line} in file {llm_file}.\n"
            f"[Diff]\n{diff}\n[/Diff]\n"
            f"[Code_Review]\n{llm_comment}\n[/Code_Review]\n"
            f"### Has this LLM generated code review violated Context Inconsistency? Answer:"
        )

        raw = single_request(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        result = _clean_load_judge_json(raw)
        verdict = False if result is None else not (result.get("True_False") == "True")
        try:
            logger = BaseEvaluator.get_json_logger()
            logger.info(
                "IsCommentContextAligned",
                extra={
                    "evaluator": "is_comment_context_aligned",
                    "diff_id": comment_dict.get("diff_id"),
                    "comment_file": llm_file,
                    "comment_line": llm_line,
                    "raw": raw,
                    "result": verdict,
                },
            )
        except FileNotFoundError:
            pass
        return verdict
