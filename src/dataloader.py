# dataloader.py
#
# Description: unified dataloader for the new evaluation pipeline.
#              Used by evaluator.py and leaderboard.py.
#
# Agent identity comes from the 'submission' block inside each llm-comments
# file — no pre-registered agents needed.
#
# Eval-results are split into two files:
#   {stem}_comments.jsonl    – one row per comment, with metric columns
#   {stem}_trajectory.jsonl  – one row per diff_id, with trajectory fields

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Registry dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRegistry:
    name: str = ""
    display_name: str = ""
    benchmark_goal: str = ""
    evaluator_classes: list = field(default_factory=list)
    is_ground_truth_eligible: bool = False
    # Per-metric aggregation modes.
    # Keys are metric names (e.g. "metric/bug/is_comment_location_relevant_matched_recall").
    # Values are one of:
    #   "precision"    – mean(bool per comment)  [default for bool metrics]
    #   "mean"         – mean(float per comment) [default for score metrics]
    #   "recall"       – covered_gt / total_gt   [requires ground truth]
    #   "sum_per_diff" – mean(per-diff sums)      [for trajectory cost/tokens]
    metric_aggregation: dict = field(default_factory=dict)
    primary_metric: str = ""
    column_groups: dict = field(default_factory=dict)
    dataset_total_diffs: int = 0
    task_accomplishment_mode: str = "submitted"  # "submitted" | "has_reviews"
    group_summary: dict = field(
        default_factory=dict
    )  # {group_name: {"method": ..., "columns": [...] | "column": ...}}
    venn_diagram: dict = field(default_factory=dict)  # Venn diagram configuration


@dataclass
class SubmissionMeta:
    """Extracted from the 'submission' block of an llm-comments file."""

    agent_id: str = ""
    agent_version: str = ""
    model: str = ""
    timestamp: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------


def load_benchmark(
    benchmark_name: str,
    benchmarks_root: str = "../benchmarks",
) -> BenchmarkRegistry:
    path = Path(benchmarks_root) / benchmark_name / "benchmark_info.json"
    _require_file(str(path))
    with open(path) as f:
        d = json.load(f)
    lb = d.get("leaderboard", {})
    return BenchmarkRegistry(
        name=d["name"],
        display_name=d.get("display_name", d["name"]),
        benchmark_goal=d["benchmark_goal"],
        evaluator_classes=d.get("evaluator_classes", []),
        is_ground_truth_eligible=d.get("is_ground_truth_eligible", False),
        metric_aggregation=d.get("metric_aggregation", {}),
        primary_metric=lb.get("primary_metric", ""),
        column_groups=lb.get("column_groups", {}),
        dataset_total_diffs=d.get("dataset_total_diffs", 0),
        task_accomplishment_mode=d.get("task_accomplishment_mode", "submitted"),
        group_summary=lb.get("group_summary", {}),
        venn_diagram=d.get("venn_diagram", {}),
    )


def load_benchmark_ground_truth(
    benchmark_name: str,
    benchmarks_root: str = "../benchmarks",
) -> pd.DataFrame | None:
    path = (
        Path(benchmarks_root) / benchmark_name / "input-dataset" / "groundtruth.jsonl"
    )
    if not path.exists():
        return None
    return pd.read_json(str(path), lines=True)


def load_benchmark_dataset(
    benchmark_name: str,
    benchmarks_root: str = "../benchmarks",
) -> pd.DataFrame | None:
    path = Path(benchmarks_root) / benchmark_name / "input-dataset" / "dataset.jsonl"
    if not path.exists():
        return None
    return pd.read_json(str(path), lines=True)


# ---------------------------------------------------------------------------
# LLM-comments loader
# ---------------------------------------------------------------------------


def load_llm_comments(file_path: str) -> tuple[SubmissionMeta, pd.DataFrame]:
    """
    Load a unified llm-comments JSONL file.

    Each line must have at minimum:
        diff_id    – identifier of the PR/diff
        submission – object with agent identity fields (MANDATORY: model field)
        trajectory – object with mandatory cost/token fields (one per diff)
        reviews    – list of comment objects, each with: file, line, comment

    Returns:
        (SubmissionMeta, flat_df)  where flat_df has one row per comment.

    The 'trajectory' dict is carried as a column on each comment row so
    evaluator.py can read it.  leaderboard.py deduplicates by diff_id
    using the separate _trajectory.jsonl file.

    The 'submission' dict is also carried as a column for evaluators that need
    access to metadata like the model name.

    Extra fields in each review object pass through as additional columns.

    Raises:
        ValueError: If model field is missing from submission block.
    """
    _require_file(file_path)

    rows: list[dict] = []
    submission_meta: SubmissionMeta | None = None

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Extract submission metadata from the first record
            if submission_meta is None:
                sub = record.get("submission", {})

                # ENFORCE: model field is mandatory
                if not sub.get("model"):
                    raise ValueError(
                        f"Submission block missing mandatory 'model' field in {file_path}. "
                        f"Model is required for trajectory cost calculations."
                    )

                submission_meta = SubmissionMeta(
                    agent_id=sub.get("agent_id", ""),
                    agent_version=sub.get("agent_version", ""),
                    model=sub.get("model", ""),
                    timestamp=sub.get("timestamp", ""),
                    extra={
                        k: v
                        for k, v in sub.items()
                        if k not in ("agent_id", "agent_version", "model", "timestamp")
                    },
                )

            diff_id = record["diff_id"]
            trajectory = record.get("trajectory", {})
            reviews = record.get("reviews", [])
            submission = record.get("submission", {})

            # If no reviews, create a single row to show task was accomplished
            # but no comments were generated (important for task accomplishment rate)
            if not reviews:
                row = {
                    "diff_id": diff_id,
                    # No comment fields for empty reviews
                    "comment_file": None,
                    "comment_line": None,
                    "comment": None,
                    # Trajectory dict carried per-row; deduplicated in leaderboard
                    "trajectory": trajectory,
                    # Submission dict carried per-row for evaluator access
                    "submission": submission,
                }
                rows.append(row)
            else:
                # Normal case: one row per review
                for review in reviews:
                    row = {
                        "diff_id": diff_id,
                        # Normalized comment fields
                        "comment_file": review.get("file"),
                        "comment_line": review.get("line"),
                        "comment": review.get("comment"),
                        # Trajectory dict carried per-row; deduplicated in leaderboard
                        "trajectory": trajectory,
                        # Submission dict carried per-row for evaluator access
                        "submission": submission,
                        # Pass through any extra review fields
                        **{
                            k: v
                            for k, v in review.items()
                            if k not in ("file", "line", "comment")
                        },
                    }
                    rows.append(row)

    if submission_meta is None:
        submission_meta = SubmissionMeta()
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    return submission_meta, df


# ---------------------------------------------------------------------------
# Eval-results loaders  (split-file format)
# ---------------------------------------------------------------------------


def load_eval_comments(file_path: str) -> pd.DataFrame:
    """Load the per-comment eval-results file ({stem}_comments.jsonl)."""
    _require_file(file_path)
    return pd.read_json(file_path, lines=True)


def load_eval_trajectory(file_path: str) -> pd.DataFrame:
    """Load the per-diff trajectory file ({stem}_trajectory.jsonl)."""
    _require_file(file_path)
    return pd.read_json(file_path, lines=True)


def find_eval_result_pair(
    benchmark_name: str,
    agent_stem: str,
    benchmarks_root: str = "../benchmarks",
) -> tuple[Path | None, Path | None]:
    """
    Return (comments_path, trajectory_path) for a given agent stem.
    Returns (None, None) if either file is missing.
    """
    base = Path(benchmarks_root) / benchmark_name / "eval-results"
    comments = base / f"{agent_stem}_comments.jsonl"
    trajectory = base / f"{agent_stem}_trajectory.jsonl"
    return (
        comments if comments.exists() else None,
        trajectory if trajectory.exists() else None,
    )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def list_eval_result_stems(
    benchmark_name: str,
    benchmarks_root: str = "../benchmarks",
) -> list[str]:
    """
    Return unique agent stems that have BOTH a _comments.jsonl AND a _trajectory.jsonl.
    A stem is the filename without the _comments/_trajectory suffix and .jsonl extension.
    """
    d = Path(benchmarks_root) / benchmark_name / "eval-results"
    if not d.exists():
        return []

    comment_stems = {
        f.name[: -len("_comments.jsonl")] for f in d.glob("*_comments.jsonl")
    }
    trajectory_stems = {
        f.name[: -len("_trajectory.jsonl")] for f in d.glob("*_trajectory.jsonl")
    }
    return sorted(comment_stems & trajectory_stems)


def list_benchmarks(benchmarks_root: str = "../benchmarks") -> list[str]:
    """Return benchmark names that have a benchmark_info.json."""
    root = Path(benchmarks_root)
    if not root.exists():
        return []
    return sorted(
        d.name
        for d in root.iterdir()
        if d.is_dir() and (d / "benchmark_info.json").exists()
    )


def parse_submission_filename(filename: str) -> dict:
    """
    Parse a submission filename of the form:
        {agent_id}_{YYYYMMDD-HHMM}[_optional_suffix].jsonl

    The timestamp segment is the first underscore-delimited part whose first 8
    characters are all digits (YYYYMMDD).

    Examples:
        sample-agent_20260101-1200.jsonl        → agent_id=sample-agent
        rovodev-cli-0-12-16_20251105-0800.jsonl → agent_id=rovodev-cli-0-12-16
        gpt4o_20250225-2255_baseline.jsonl       → agent_id=gpt4o
        my-agent_20260301-0900_run2.jsonl        → agent_id=my-agent

    Returns dict with keys: agent_id, timestamp (may be empty string).
    """
    stem = Path(filename).stem  # strip .jsonl / _comments.jsonl / _trajectory.jsonl
    # Also strip _comments / _trajectory suffixes if present
    for suffix in ("_comments", "_trajectory"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    parts = stem.split("_")

    # Find first part whose first 8 chars look like YYYYMMDD
    ts_idx = None
    for i, p in enumerate(parts):
        if re.match(r"^\d{8}", p):
            ts_idx = i
            break

    if ts_idx is not None and ts_idx > 0:
        agent_id = "_".join(parts[:ts_idx])
        timestamp = "_".join(parts[ts_idx:])
    else:
        agent_id = stem
        timestamp = ""

    return {"agent_id": agent_id, "timestamp": timestamp}
