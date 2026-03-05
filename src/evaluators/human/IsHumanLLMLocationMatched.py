# IsHumanLLMLocationMatched.py
#
# Description: Does an LLM comment land within ±LINE_WINDOW lines of any
#              human ground-truth comment on the same file?
#              Precision metric: evaluated per LLM comment.

from ..base import BaseEvaluator

LINE_WINDOW = 5


def _is_comment_within_line_range(comment_line: int, ground_truth_line: int) -> bool:
    return abs(comment_line - ground_truth_line) <= LINE_WINDOW


class IsHumanLLMLocationMatched(BaseEvaluator):
    evaluation_name = "metric/human/is_human_llm_location_matched"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        llm_file = comment_dict.get("comment_file")
        llm_line = comment_dict.get("comment_line")

        if llm_file is None or llm_line is None or not ground_truth:
            return False

        llm_line = int(llm_line)
        return any(
            gt.get("comment_file") == llm_file
            and _is_comment_within_line_range(llm_line, int(gt.get("comment_line", 0)))
            for gt in ground_truth
        )
