# IsCommentLocationSuggestionMatched.py
#
# Description: Composite: IsBugSuggestionValid AND IsBugLocationMatched.
#              Must be listed AFTER those two in benchmark_info.json.

from ..base import BaseEvaluator


class IsCommentLocationSuggestionMatched(BaseEvaluator):
    evaluation_name = "metric/bug/is_comment_location_suggestion_matched"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        return bool(
            comment_dict.get("metric/bug/is_bug_suggestion_valid")
            and comment_dict.get("metric/bug/is_bug_location_matched")
        )
