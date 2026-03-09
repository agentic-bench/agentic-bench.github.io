# leaderboard.py
#
# Description: Aggregate eval-results into static JSON files for leaderboard/.
#
# Usage:
#   cd src
#   python leaderboard.py [--benchmarks-root ../benchmarks] \
#                              [--output-dir ../leaderboard]
#
# Reads eval-results split files:
#   {stem}_comments.jsonl    – per-comment metric values
#   {stem}_trajectory.jsonl  – per-diff trajectory fields
#
# For each benchmark, the latest eval-results stem per agent_id is used.
#
# Outputs (written to leaderboard/data/):
#   data_{benchmark}.json        – one leaderboard entry per agent
#   output_filelist.json         – list of data_*.json filenames
#   statistics.json              – summary counts
#   benchmark_meta.json          – display names, column groups, primary metric (JS replica)
#   metric_display_names.json    – flat key→display-name map for JS

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataloader import (
    BenchmarkRegistry,
    load_benchmark,
    load_benchmark_ground_truth,
    load_eval_comments,
    load_eval_trajectory,
    list_benchmarks,
    list_eval_result_stems,
    parse_submission_filename,
    find_eval_result_pair,
)


# ---------------------------------------------------------------------------
# Aggregation modes
# ---------------------------------------------------------------------------
#
# "precision"    – mean(bool per comment, skip null)
#                  Answers: "what fraction of LLM comments satisfy this criterion?"
#                  Use for: all binary per-comment metrics (location match, relevance,
#                           alignment, informative, context-aligned, diff-relevant, etc.)
#
# "mean"         – mean(float per comment, skip null)
#                  Answers: "what is the average score per LLM comment?"
#                  Use for: continuous score metrics (ROUGE-1, ROUGE-L, BLEU, edit-sim)
#
# "recall"       – covered_gt_entries / total_gt_entries
#                  A gt entry is "covered" if ≥1 LLM comment for that diff is True.
#                  Capped per diff so one True comment covers at most one gt entry.
#                  Answers: "what fraction of human/bug ground-truth items were caught?"
#                  Use for: *_recall metrics (requires ground truth)
#
# "sum_per_diff" – mean(per-diff sums across all diffs)
#                  First sums the metric within each diff, then averages across diffs.
#                  Answers: "what is the average per-diff total for this quantity?"
#                  Use for: trajectory costs/tokens (one trajectory per diff, but metric
#                           column is repeated on every comment row for convenience).
#                           NOTE: leaderboard reads trajectory from _trajectory.jsonl,
#                           so this mode is only relevant if a metric evaluator
#                           writes a per-comment cost column.
#
# DEFAULT: bool metrics → "precision", float/score metrics → "mean"
# Override per metric in benchmark_info.json → "metric_aggregation"

AGGREGATION_MODES = {"precision", "mean", "recall", "sum_per_diff"}


def _default_aggregation(metric_col: str) -> str:
    """Infer default aggregation mode from metric name when not explicitly configured."""
    if "_score" in metric_col:
        return "mean"
    return "precision"


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: list) -> float | None:
    valid = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
    ]
    return float(np.mean(valid)) if valid else None


def _safe_sum(values: list) -> float | None:
    valid = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))
    ]
    return float(np.sum(valid)) if valid else None


def _normalize(obj: dict) -> dict:
    """Replace NaN / Inf / numpy scalars with JSON-serializable equivalents."""
    out = {}
    for k, v in obj.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out[k] = None
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, np.bool_):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Per-metric aggregation
# ---------------------------------------------------------------------------


def _aggregate_precision(eval_df: pd.DataFrame, col: str) -> float | None:
    """mean(bool per comment, skip null) → precision rate in [0, 1]."""
    if col not in eval_df.columns:
        return None
    try:
        series = eval_df[col].astype("boolean")
        valid = series.dropna()
        return float(valid.mean()) if len(valid) > 0 else None
    except Exception:
        return None


def _aggregate_mean(eval_df: pd.DataFrame, col: str) -> float | None:
    """mean(float per comment, skip null)."""
    if col not in eval_df.columns:
        return None
    try:
        series = eval_df[col].astype("float64")
        valid = series.dropna()
        return float(valid.mean()) if len(valid) > 0 else None
    except Exception:
        return None


def _aggregate_recall(
    eval_df: pd.DataFrame,
    col: str,
    gt_df: pd.DataFrame | None,
) -> float | None:
    """
    recall = covered_gt_entries / total_gt_entries

    For each diff_id in the ground truth:
      - gt_count  = number of ground-truth entries for this diff
      - true_count = number of LLM comments that are True for this diff
      - covered   += min(true_count, gt_count)   (one True = one gt covered)

    recall = sum(covered) / sum(gt_count)
    """
    if gt_df is None or gt_df.empty or col not in eval_df.columns:
        return None

    total_gt = len(gt_df)
    if total_gt == 0:
        return None

    covered = 0
    for diff_id in gt_df["diff_id"].unique():
        diff_rows = eval_df[eval_df["diff_id"] == diff_id]
        if diff_rows.empty:
            continue
        gt_count = int((gt_df["diff_id"] == diff_id).sum())
        try:
            true_count = int(diff_rows[col].astype("boolean").sum(skipna=True))
        except Exception:
            true_count = 0
        covered += min(true_count, gt_count)

    return covered / total_gt


def _aggregate_sum_per_diff(eval_df: pd.DataFrame, col: str) -> float | None:
    """
    For each diff_id: sum the metric values.
    Return the mean of those per-diff sums.
    Used for per-comment cost columns written by ops evaluators.
    (Trajectory fields from _trajectory.jsonl are handled separately.)
    """
    if col not in eval_df.columns:
        return None
    try:
        per_diff = eval_df.groupby("diff_id")[col].apply(
            lambda s: s.astype("float64").sum(skipna=True)
        )
        valid = per_diff.dropna()
        return float(valid.mean()) if len(valid) > 0 else None
    except Exception:
        return None


def _aggregate_metric(
    eval_df: pd.DataFrame,
    col: str,
    mode: str,
    gt_df: pd.DataFrame | None,
) -> float | None:
    """Dispatch to the correct aggregation function."""
    if mode == "precision":
        return _aggregate_precision(eval_df, col)
    elif mode == "mean":
        return _aggregate_mean(eval_df, col)
    elif mode == "recall":
        return _aggregate_recall(eval_df, col, gt_df)
    elif mode == "sum_per_diff":
        return _aggregate_sum_per_diff(eval_df, col)
    else:
        raise ValueError(f"Unknown aggregation mode '{mode}' for metric '{col}'.")


# ---------------------------------------------------------------------------
# Trajectory aggregation  (from _trajectory.jsonl)
# ---------------------------------------------------------------------------

TRAJECTORY_FIELDS = [
    "input_tokens",
    "output_tokens",
    "steps",
    # Cost metrics calculated by TrajectoryCostMetrics evaluator
    "trajectory_input_costs",
    "trajectory_output_costs",
    "trajectory_total_costs",
]


def _aggregate_trajectory(traj_df: pd.DataFrame) -> dict:
    """
    Aggregate trajectory fields across all diffs.

    Each row in traj_df is one diff_id.  We take the mean across diffs for
    all standard trajectory fields (tokens/costs/steps per diff).
    This tells the leaderboard: "on average, how much did each diff cost?"

    Fields not in TRAJECTORY_FIELDS are ignored.
    """
    # Flatten the nested "trajectory" dict column if present
    if "trajectory" in traj_df.columns and "input_tokens" not in traj_df.columns:
        try:
            flat = pd.json_normalize(traj_df["trajectory"].dropna().tolist())
            # Re-index to align with traj_df
            flat.index = traj_df[traj_df["trajectory"].notna()].index
            traj_df = traj_df.drop(columns=["trajectory"]).join(flat, how="left")
        except Exception:
            pass

    result = {}
    for field in TRAJECTORY_FIELDS:
        if field not in traj_df.columns:
            result[f"trajectory/{field}"] = None
            continue
        try:
            valid = traj_df[field].astype("float64").dropna()
            result[f"trajectory/{field}"] = (
                float(valid.mean()) if len(valid) > 0 else None
            )
        except Exception:
            result[f"trajectory/{field}"] = None

    return result


# ---------------------------------------------------------------------------
# Primary (overall) score
# ---------------------------------------------------------------------------


def _compute_overall_score(
    metrics: dict,
    benchmark: BenchmarkRegistry,
) -> float | None:
    """
    Compute an overall weighted score.

    If benchmark.leaderboard.primary_metric is defined, return that metric's value.
    Otherwise return the mean of all non-None metric values.
    """
    primary = benchmark.primary_metric
    if primary and primary in metrics and metrics[primary] is not None:
        return metrics[primary]
    # Fallback: mean of all metric values
    valid = [
        v for v in metrics.values() if isinstance(v, (int, float)) and not np.isnan(v)
    ]
    return float(np.mean(valid)) if valid else None


# ---------------------------------------------------------------------------
# Per-benchmark aggregation
# ---------------------------------------------------------------------------


def _aggregate_benchmark(
    benchmark: BenchmarkRegistry,
    stems: list[str],
    gt_df: pd.DataFrame | None,
    benchmarks_root: str,
    dir_name: (
        str | None
    ) = None,  # physical directory name (may differ from benchmark.name)
) -> list[dict]:
    """
    For each agent stem (latest per agent_id):
      1. Load _comments.jsonl → aggregate metrics
      2. Load _trajectory.jsonl → aggregate trajectory fields
      3. Merge into one leaderboard row
    """
    # Resolve latest stem per agent_id
    latest: dict[str, str] = {}  # agent_id → latest stem
    for stem in sorted(stems):
        parsed = parse_submission_filename(stem + ".jsonl")
        aid = parsed["agent_id"]
        # Stems are sorted by name; later = newer timestamp → overwrite
        latest[aid] = stem

    results = []
    for agent_id, stem in sorted(latest.items()):
        bench_dir = dir_name or benchmark.name
        comments_path, traj_path = find_eval_result_pair(
            bench_dir, stem, benchmarks_root
        )
        if comments_path is None or traj_path is None:
            print(f"  [WARN] Missing eval files for stem '{stem}', skipping.")
            continue

        eval_df = load_eval_comments(str(comments_path))
        traj_df = load_eval_trajectory(str(traj_path))

        # Extract submission metadata from first row
        model = ""
        timestamp = ""
        if not eval_df.empty:
            first = eval_df.iloc[0]
            # Try to extract model from submission dict (new format)
            if "submission" in first and isinstance(first["submission"], dict):
                model = str(first["submission"].get("model", "")) or ""
                timestamp = str(first["submission"].get("timestamp", "")) or ""
            # Fallback to old _model format
            else:
                model = str(first.get("_model", "")) or ""
            timestamp = parse_submission_filename(stem + ".jsonl")["timestamp"]

        total_diffs = int(eval_df["diff_id"].nunique()) if not eval_df.empty else 0
        # Count diffs where agent produced at least one review (task accomplished)
        accomplished_diffs = 0
        mode = benchmark.task_accomplishment_mode
        if not traj_df.empty:
            if mode == "has_reviews" and "has_reviews" in traj_df.columns:
                # Bug-style: must have produced at least one review
                accomplished_diffs = int(traj_df["has_reviews"].sum())
            else:
                # Submitted-style (contextcrbench): any submitted diff = accomplished
                accomplished_diffs = int(traj_df["diff_id"].nunique())
        else:
            accomplished_diffs = total_diffs
        dataset_total = benchmark.dataset_total_diffs or total_diffs
        task_accomplishment_rate = (
            accomplished_diffs / dataset_total if dataset_total > 0 else None
        )
        total_comments = int(len(eval_df))

        # Aggregate all metric/* columns present in the eval file
        metric_results = {}
        metric_cols = [c for c in eval_df.columns if c.startswith("metric/")]
        agg_cfg = benchmark.metric_aggregation

        for col in metric_cols:
            mode = agg_cfg.get(col, _default_aggregation(col))
            val = _aggregate_metric(eval_df, col, mode, gt_df)
            metric_results[col] = val

        # Aggregate trajectory fields from the trajectory file
        traj_results = _aggregate_trajectory(traj_df)

        # Compute overall score
        primary_metric = benchmark.primary_metric
        if primary_metric and primary_metric in metric_results:
            overall_score = metric_results[primary_metric]
        else:
            valid_vals = [
                v
                for v in metric_results.values()
                if isinstance(v, (int, float)) and not np.isnan(float(v))
            ]
            overall_score = float(np.mean(valid_vals)) if valid_vals else None

        # ================================================================== #
        # GROUP SCORE AGGREGATION FOR COLLAPSED COLUMN DISPLAY
        # ================================================================== #
        # When users collapse column groups in the leaderboard, a single
        # "group_score/{group_name}" column is shown instead of all individual
        # metrics in that group.
        #
        # This aggregation is computed here in Python and stored in the JSON:
        #   - Aggregation rules are defined in benchmark_info.json
        #   - Under leaderboard → group_summary → {group_name}
        #   - The computed value is stored as group_score/{group_name}
        #   - The JS reads this and displays it when the group is collapsed
        #
        # AGGREGATION METHODS:
        #   "mean"  → arithmetic mean of listed metric columns
        #             e.g., Code Review Capability = mean(human_aligned, context_aligned, location_matched)
        #   "pick"  → value of a single named column
        #             e.g., Trajectory = trajectory/total_costs
        #
        # TODO: Update column collapsing aggregation logic
        # ================================================================== #
        all_values = {**metric_results, **traj_results}
        group_scores: dict = {}
        for group_name, cfg in benchmark.group_summary.items():
            key = f"group_score/{group_name}"
            method = cfg.get("method", "mean")
            if method == "mean":
                # Compute mean of all non-null values in the specified columns
                cols = cfg.get("columns", [])
                vals = [
                    all_values[c]
                    for c in cols
                    if c in all_values and all_values[c] is not None
                ]
                group_scores[key] = float(sum(vals) / len(vals)) if vals else None
            elif method == "pick":
                # Pick a single column value (used for trajectory costs)
                col = cfg.get("column", "")
                group_scores[key] = all_values.get(col)

        row = _normalize(
            {
                "agent": agent_id,
                "model": model,
                "timestamp": timestamp,
                "benchmark": benchmark.name,
                "benchmark_goal": benchmark.benchmark_goal,
                "dataset_total_diffs": dataset_total,
                "accomplished_diffs": accomplished_diffs,
                "task_accomplishment_rate": task_accomplishment_rate,
                "total_diffs": total_diffs,
                "total_comments": total_comments,
                "overall_weighted_score": overall_score,
                **group_scores,
                **metric_results,
                **traj_results,
            }
        )
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Benchmark meta (JS-accessible replica of benchmark_info.json leaderboard cfg)
# ---------------------------------------------------------------------------


def _sorted_benchmarks(benchmarks_root: str) -> list[str]:
    """Return benchmark dir names sorted by tab_order field in benchmark_info.json."""

    def _order(bname):
        info_path = Path(benchmarks_root) / bname / "benchmark_info.json"
        try:
            d = json.load(open(info_path))
            return d.get("tab_order", 999)
        except Exception:
            return 999

    return sorted(list_benchmarks(benchmarks_root), key=_order)


def _build_benchmark_meta(benchmarks_root: str) -> dict:
    """
    Read benchmark_info.json for every benchmark and build a JS-friendly summary.
    This is the only file the HTML/JS reads — benchmark_info.json itself is
    Python-pipeline-only.

    Includes:
      - display_name, benchmark_goal, description
      - primary_metric, column_groups
      - metric_aggregation  (so JS can format % vs raw correctly)
      - display_names       (per-benchmark overrides, merged with builtins by JS)
    """
    meta = {}
    for bname in _sorted_benchmarks(benchmarks_root):
        info_path = Path(benchmarks_root) / bname / "benchmark_info.json"
        with open(info_path) as f:
            d = json.load(f)
        lb = d.get("leaderboard", {})
        # Use logical name from benchmark_info.json (may differ from dir name)
        logical_name = d.get("name", bname)
        meta[logical_name] = {
            "display_name": d.get("display_name", bname),
            "benchmark_goal": d.get("benchmark_goal", ""),
            "description": d.get("description", ""),
            "primary_metric": lb.get("primary_metric", ""),
            "column_groups": lb.get("column_groups", {}),
            # group_summary: one representative column per group shown when collapsed.
            # Keys = group names; each value has "method" ("mean"|"pick") and
            # "columns" (list) or "column" (single key).
            # Python stores computed values as group_score/{group_name} in data JSON.
            # JS reads this to know which group_score key to display per group.
            "group_summary": lb.get("group_summary", {}),
            # Aggregation modes: JS uses this to decide % vs raw number formatting
            "metric_aggregation": d.get("metric_aggregation", {}),
            # Per-benchmark display name overrides (merged with builtins in JS)
            "display_names": lb.get("display_names", {}),
            "dataset_total_diffs": d.get("dataset_total_diffs", 0),
            "benchmark_goal": d.get("benchmark_goal", ""),
        }
    return meta


# ---------------------------------------------------------------------------
# Display names (flat map: metric_key → human label)
# Populated dynamically from benchmark_info.json leaderboard.display_names,
# with hardcoded fallbacks for standard fields.
# ---------------------------------------------------------------------------

_BUILTIN_DISPLAY_NAMES = {
    # Identity columns
    "agent": "Agent",
    "model": "Base Model",
    "benchmark_goal": "Benchmark",
    "timestamp": "Submission",
    "dataset_total_diffs": "Dataset PRs",
    "accomplished_diffs": "Accomplished",
    "task_accomplishment_rate": "Task Rate",
    "total_diffs": "Submitted PRs",
    "total_comments": "Total Comments",
    "overall_weighted_score": "Overall Score",
    # Human-alignment metrics
    "metric/human/is_human_llm_location_matched": "LLM Localization (Precision)",
    "metric/human/is_human_comment_location_matched_recall": "Human Comment Coverage (Recall)",
    "metric/human/llm_comment_rouge1_score": "ROUGE-1",
    "metric/human/llm_comment_rougel_score": "ROUGE-L",
    "metric/human/llm_comment_bleu_score": "BLEU",
    "metric/human/llm_comment_edit_similarity_score": "Edit Similarity",
    "metric/human/is_llm_human_aligned": "Human Alignment",
    "metric/human/is_llm_context_aligned": "Context Alignment",
    # Bug capacity metrics
    "metric/bug/is_bug_location_matched": "Location Match (Precision)",
    "metric/bug/is_bug_comment_relevant": "Comment Relevance (Precision)",
    "metric/bug/is_bug_suggestion_valid": "Suggestion Quality (Precision)",
    "metric/bug/is_comment_location_relevant_matched": "Precision (Loc+Rel)",
    "metric/bug/is_comment_location_suggestion_matched": "Precision (Loc+Sug)",
    "metric/bug/is_comment_location_relevant_matched_recall": "Recall (Loc+Rel)",
    # Judge metrics
    "metric/judge/is_comment_context_aligned": "Context Consistency",
    "metric/judge/is_comment_informative": "Informativeness",
    "metric/judge/is_comment_diff_relevant": "Diff Relevance",
    # Ops metrics
    "metric/ops/total_cost": "Op. Cost (USD)",
    # Trajectory fields
    "trajectory/input_tokens": "Input Tokens / Diff",
    "trajectory/output_tokens": "Output Tokens / Diff",
    "trajectory/steps": "Steps / Diff",
    "trajectory/trajectory_input_costs": "Input Cost / Diff (USD)",
    "trajectory/trajectory_output_costs": "Output Cost / Diff (USD)",
    "trajectory/trajectory_total_costs": "Total Cost / Diff (USD)",
}


def _build_display_names(benchmarks_root: str) -> dict:
    """
    Merge built-in display names with per-benchmark overrides from
    benchmark_info.json → leaderboard.display_names.

    Priority: per-benchmark overrides > built-in defaults.
    The JS also does a second-pass merge using benchmark_meta[b].display_names
    so per-benchmark names are always applied when rendering that benchmark's tab.
    """
    names = dict(_BUILTIN_DISPLAY_NAMES)
    for bname in _sorted_benchmarks(benchmarks_root):
        info_path = Path(benchmarks_root) / bname / "benchmark_info.json"
        with open(info_path) as f:
            d = json.load(f)
        overrides = d.get("leaderboard", {}).get("display_names", {})
        names.update(overrides)
    return names


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def update_leaderboard(
    benchmarks_root: str = "benchmarks",
    output_dir: str = "leaderboard",
):
    output_path = Path(output_dir) / "data"
    output_path.mkdir(parents=True, exist_ok=True)

    benchmarks = _sorted_benchmarks(benchmarks_root)
    if not benchmarks:
        print(f"No benchmarks found in '{benchmarks_root}'.")
        return

    print(f"\n=== Updating leaderboard ===")
    print(f"  Benchmarks root : {benchmarks_root}")
    print(f"  Output dir      : {output_dir}")
    print(f"  Benchmarks found: {benchmarks}")

    output_filelist = []
    all_agents: set[str] = set()
    all_models: set[str] = set()
    total_diffs = 0
    total_comments = 0

    for bname in benchmarks:
        print(f"\n-- Benchmark: {bname} --")
        benchmark = load_benchmark(bname, benchmarks_root=benchmarks_root)
        gt_df = load_benchmark_ground_truth(bname, benchmarks_root=benchmarks_root)

        stems = list_eval_result_stems(bname, benchmarks_root=benchmarks_root)
        if not stems:
            print(f"  No eval-result files found, skipping.")
            continue

        print(f"  Stems found: {stems}")

        agents_perf = _aggregate_benchmark(
            benchmark, stems, gt_df, benchmarks_root, dir_name=bname
        )

        for ap in agents_perf:
            all_agents.add(ap.get("agent", ""))
            all_models.add(ap.get("model", ""))
            total_diffs += ap.get("total_diffs", 0)
            total_comments += ap.get("total_comments", 0)

        out_filename = f"data_{benchmark.name}.json"
        out_file = output_path / out_filename
        with open(out_file, "w") as f:
            json.dump(agents_perf, f, indent=2)
        print(f"  Written: {out_file}  ({len(agents_perf)} agent rows)")
        output_filelist.append(out_filename)

    # output_filelist.json
    with open(output_path / "output_filelist.json", "w") as f:
        json.dump(output_filelist, f, indent=2)

    # statistics.json
    with open(output_path / "statistics.json", "w") as f:
        json.dump(
            {
                "total_agents": len(all_agents),
                "total_models": len(all_models),
                "total_reviews": total_diffs,
                "total_generated_comments": total_comments,
            },
            f,
            indent=2,
        )

    # benchmark_meta.json  (read by JS — replica of benchmark_info leaderboard config)
    with open(output_path / "benchmark_meta.json", "w") as f:
        json.dump(_build_benchmark_meta(benchmarks_root), f, indent=2)

    # metric_display_names.json  (flat key→label lookup for JS)
    with open(output_path / "metric_display_names.json", "w") as f:
        json.dump(_build_display_names(benchmarks_root), f, indent=2)

    print(f"\n=== Leaderboard updated ===")
    print(f"  Benchmarks : {len(output_filelist)}")
    print(f"  Agents     : {len(all_agents)}")
    print(f"  Models     : {len(all_models)}")
    print(f"  Diffs      : {total_diffs}")
    print(f"  Comments   : {total_comments}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate leaderboard static JSON files from eval-results."
    )
    parser.add_argument(
        "--benchmarks-root",
        default="benchmarks",
        help="Root folder for benchmarks (default: benchmarks)",
    )
    parser.add_argument(
        "--output-dir",
        default="leaderboard",
        help="Output folder for leaderboard static files (default: leaderboard)",
    )
    args = parser.parse_args()
    update_leaderboard(
        benchmarks_root=args.benchmarks_root,
        output_dir=args.output_dir,
    )
