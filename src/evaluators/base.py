# base.py
#
# Description: base evaluator class for the new evaluation pipeline.
#              All evaluators receive a flat comment dict and a list of
#              ground-truth dicts keyed by diff_id.
#
#              The comment dict has the following guaranteed keys:
#                  diff_id        – identifier of the PR/diff
#                  comment_file   – file path the comment was placed on
#                  comment_line   – integer line number
#                  comment        – text of the LLM comment
#                  trajectory     – dict with mandatory trajectory fields
#              Plus any extra fields from the review object (passed through).
#
#              evaluate() must return: bool | float | None
#                  None  → not applicable / evaluator could not run

import logging
import os

from pythonjsonlogger import jsonlogger


class BaseEvaluator:
    evaluation_name = "BaseEvaluator"

    LOG_ENV_VAR = "EVALUATION_LOG_FILE_PATH"

    @staticmethod
    def evaluate(comment_dict: dict, ground_truth: list[dict]) -> "bool | float | None":
        raise NotImplementedError("Subclasses must implement evaluate().")

    # ------------------------------------------------------------------
    # Logging helpers (unchanged from original)
    # ------------------------------------------------------------------

    @staticmethod
    def get_current_log_filename():
        return os.environ.get(BaseEvaluator.LOG_ENV_VAR)

    @staticmethod
    def get_json_logger():
        log_filename = BaseEvaluator.get_current_log_filename()
        if not log_filename:
            raise FileNotFoundError("Evaluation log not defined by evaluator.py")

        logger = logging.getLogger(log_filename)
        if not logger.handlers:
            handler = logging.FileHandler(log_filename)
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
