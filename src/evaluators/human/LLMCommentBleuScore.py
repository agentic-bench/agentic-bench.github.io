# LLMCommentBleuScore.py
#
# Description: SacreBLEU score of LLM comment vs closest in-window human comment.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sacrebleu
from ..base import BaseEvaluator

LINE_WINDOW = 5


class LLMCommentBleuScore(BaseEvaluator):
    evaluation_name = "metric/human/llm_comment_bleu_score"

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

        references = [gt.get("comment_content", "") for gt in aligned]
        bleu = sacrebleu.corpus_bleu([llm_content], [references])
        # normalize to [0, 1]
        return bleu.score / 100.0
