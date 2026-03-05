# IsHumanCommentLocationMatched.py
#
# Description: Recall-oriented variant: does this LLM comment cover any human
#              ground-truth comment?  Used together with recall aggregation in
#              leaderboard.py (metric_aggregation: "recall").

from .IsHumanLLMLocationMatched import _is_comment_within_line_range
from ..base import BaseEvaluator


class IsHumanCommentLocationMatched(BaseEvaluator):
    # NOTE: metric name ends with _recall → leaderboard uses recall aggregation
    evaluation_name = "metric/human/is_human_comment_location_matched_recall"

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
