# IsBugLocationMatched.py
#
# Description: evaluate comment location against bug ground truth.
#              Uses a ±LINE_WINDOW fuzzy window around the ground-truth line.
#              Ground truth supports both single-line (end_line == -1) and
#              range entries.

from ..base import BaseEvaluator

LINE_WINDOW = 5


def _line_within_window(llm_line: int, gt_start: int, gt_end: int) -> bool:
    """
    Return True when llm_line falls within [gt_start - WINDOW, gt_end + WINDOW].
    When gt_end == -1 the ground truth is a single line (gt_start).
    """
    effective_end = gt_start if gt_end == -1 else gt_end
    return (gt_start - LINE_WINDOW) <= llm_line <= (effective_end + LINE_WINDOW)


class IsBugLocationMatched(BaseEvaluator):
    evaluation_name = "metric/bug/is_bug_location_matched"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> bool:
        llm_file = comment_dict.get("comment_file")
        llm_line = comment_dict.get("comment_line")

        if llm_file is None or llm_line is None or not ground_truth:
            return False

        llm_line = int(llm_line)

        return any(
            gt.get("file_path") == llm_file
            and _line_within_window(
                llm_line,
                int(gt.get("start_line", 0)),
                int(gt.get("end_line", -1)),
            )
            for gt in ground_truth
        )
