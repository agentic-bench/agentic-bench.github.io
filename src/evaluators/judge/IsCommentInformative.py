# IsCommentInformative.py
#
# Description: LLM judge — is the comment informative?
#              Scores: 0 = generic, 1 = vague, 2 = concrete.
#              Returns True when score >= 1 (not purely generic).

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.direct_gateway import single_request
from ..base import BaseEvaluator

MODEL = "gpt-4.1-2025-04-14"

SYSTEM_PROMPT = """\
You are an expert software engineer evaluating the quality of LLM-generated code review comments.

Classify the comment into one of three categories:
- "generic": the comment is too vague or generic to be useful (score 0)
- "vague": the comment identifies an issue but lacks specific detail (score 1)
- "concrete": the comment is specific, actionable, and well-reasoned (score 2)

Respond with a JSON object: {"label": "<generic|vague|concrete>", "score": <0|1|2>}"""

USER_PROMPT_TEMPLATE = (
    "Evaluate the following LLM-generated code review comment:\n\n"
    "[Comment]\n{comment}\n[/Comment]\n\n"
    "Respond with JSON: {{\"label\": \"<generic|vague|concrete>\", \"score\": <0|1|2>}}"
)


class IsCommentInformative(BaseEvaluator):
    evaluation_name = "metric/judge/is_comment_informative"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        llm_comment = comment_dict.get("comment", "")
        if not llm_comment:
            return False

        raw = single_request(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=USER_PROMPT_TEMPLATE.format(comment=llm_comment),
            json_output=True,
        )
        try:
            result = json.loads(raw) if isinstance(raw, str) else raw
            score = int(result.get("score", 0))
            verdict = score >= 1
        except Exception:
            result = {}
            score = 0
            verdict = False
        try:
            logger = BaseEvaluator.get_json_logger()
            logger.info("IsCommentInformative", extra={
                "evaluator": "is_comment_informative",
                "diff_id": comment_dict.get("diff_id"),
                "label": result.get("label") if isinstance(result, dict) else None,
                "score": score,
                "result": verdict,
            })
        except FileNotFoundError:
            pass
        return verdict
