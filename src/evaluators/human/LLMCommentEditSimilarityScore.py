# LLMCommentEditSimilarityScore.py
#
# Description: Normalized edit-distance similarity of LLM comment vs closest
#              in-window human comment (1 - normalized_edit_distance).

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import editdistance
from ..base import BaseEvaluator

LINE_WINDOW = 5


def _normalized_similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - editdistance.eval(a, b) / max_len


class LLMCommentEditSimilarityScore(BaseEvaluator):
    evaluation_name = "metric/human/llm_comment_edit_similarity_score"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> "float | None":
        if not ground_truth:
            return None

        llm_content = comment_dict.get("comment", "")
        llm_file = comment_dict.get("comment_file", "")
        llm_line = int(comment_dict.get("comment_line", 0))

        aligned = [
            gt for gt in ground_truth
            if gt.get("comment_file") == llm_file
            and abs(llm_line - int(gt.get("comment_line", 0))) <= LINE_WINDOW
        ]
        if not aligned:
            return None

        scores = [
            _normalized_similarity(llm_content, gt.get("comment_content", ""))
            for gt in aligned
        ]
        return max(scores)
