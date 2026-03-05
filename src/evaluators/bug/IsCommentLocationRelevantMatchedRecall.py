# IsCommentLocationRelevantMatchedRecall.py
#
# Description: Same composite as IsCommentLocationRelevantMatched but tagged
#              for recall aggregation (leaderboard divides by ground-truth count
#              when metric_aggregation maps this name to "recall").
#              Must be listed AFTER IsBugCommentRelevant and IsBugLocationMatched.

from ..base import BaseEvaluator


class IsCommentLocationRelevantMatchedRecall(BaseEvaluator):
    # NOTE: metric name ends with _recall → leaderboard uses recall aggregation
    evaluation_name = "metric/bug/is_comment_location_relevant_matched_recall"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        return bool(
            comment_dict.get("metric/bug/is_bug_comment_relevant")
            and comment_dict.get("metric/bug/is_bug_location_matched")
        )
