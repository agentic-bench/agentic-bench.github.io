# IsCommentLocationRelevantMatched.py
#
# Description: Composite: IsBugCommentRelevant AND IsBugLocationMatched.
#              Must be listed AFTER those two in benchmark_info.json.

from ..base import BaseEvaluator


class IsCommentLocationRelevantMatched(BaseEvaluator):
    evaluation_name = "metric/bug/is_comment_location_relevant_matched"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        return bool(
            comment_dict.get("metric/bug/is_bug_comment_relevant")
            and comment_dict.get("metric/bug/is_bug_location_matched")
        )
