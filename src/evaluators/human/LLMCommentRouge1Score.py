# LLMCommentRouge1Score.py
#
# Description: ROUGE-1 F-score of the LLM comment vs the closest (in-window)
#              human ground-truth comment on the same file.
#              Returns None when no aligned ground truth exists (excluded from mean).

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from rouge_score import rouge_scorer
from ..base import BaseEvaluator

LINE_WINDOW = 5


class LLMCommentRouge1Score(BaseEvaluator):
    evaluation_name = "metric/human/llm_comment_rouge1_score"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> "float | None":
        if not ground_truth:
            return None

        llm_content = comment_dict.get("comment", "")
        llm_file = comment_dict.get("comment_file", "")
        llm_line = int(comment_dict.get("comment_line", 0))

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        aligned = [
            gt for gt in ground_truth
            if gt.get("comment_file") == llm_file
            and abs(llm_line - int(gt.get("comment_line", 0))) <= LINE_WINDOW
        ]
        if not aligned:
            return None

        scores = [
            scorer.score(llm_content, gt.get("comment_content", ""))["rouge1"].fmeasure
            for gt in aligned
        ]
        return max(scores)
