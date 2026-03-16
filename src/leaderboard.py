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
# Venn diagram data computation
# ---------------------------------------------------------------------------


def _evaluate_expression_on_df(expr: str, df: pd.DataFrame) -> pd.Series:
    """
    Evaluate a metric expression at the comment level.
    
    Args:
        expr: Expression like "and(metric1, metric2)" or simple "metric1"
        df: DataFrame with comment-level metric columns
    
    Returns:
        Boolean Series indicating True/False for each comment
    """
    import re
    
    # Base case: simple metric column
    if not any(op in expr for op in ['and(', 'or(', 'not(']):
        if expr in df.columns:
            try:
                return df[expr].astype("boolean", errors="ignore").fillna(False)
            except Exception:
                return pd.Series([False] * len(df), index=df.index)
        return pd.Series([False] * len(df), index=df.index)
    
    # Parse and(metric1, metric2)
    match_and = re.match(r'^and\((.*?),\s*(.*?)\)$', expr.strip())
    if match_and:
        left = _evaluate_expression_on_df(match_and.group(1), df)
        right = _evaluate_expression_on_df(match_and.group(2), df)
        return left & right
    
    # Parse or(metric1, metric2)
    match_or = re.match(r'^or\((.*?),\s*(.*?)\)$', expr.strip())
    if match_or:
        left = _evaluate_expression_on_df(match_or.group(1), df)
        right = _evaluate_expression_on_df(match_or.group(2), df)
        return left | right
    
    # Parse not(metric)
    match_not = re.match(r'^not\((.*?)\)$', expr.strip())
    if match_not:
        val = _evaluate_expression_on_df(match_not.group(1), df)
        return ~val
    
    # Fallback: return all False
    return pd.Series([False] * len(df), index=df.index)


def _compute_venn_diagram_data(
    benchmark: BenchmarkRegistry,
    agents_perf: list[dict],
    eval_results_dir: Path,
    benchmarks_root: str,
) -> dict | None:
    """
    Compute Venn diagram data showing diff_id overlap detected by top N agents.

    For each of the top N agents (by primary_metric score):
    1. Read their eval-results/*_comments.jsonl
    2. Find diff_ids where at least 1 comment has primary_metric = True/1
    3. Return sets for Venn diagram visualization

    Returns:
    {
        "agents": ["agent1", "agent2", "agent3"],
        "sets": {
            "agent1": ["diff_id_1", "diff_id_3", ...],
            "agent2": ["diff_id_2", "diff_id_3", ...],
            "agent3": ["diff_id_1", "diff_id_2", ...]
        },
        "intersections": {
            "all_3": 42,
            "agent1_agent2": 15,
            "agent1_agent3": 20,
            "agent2_agent3": 18,
            "agent1_only": 5,
            "agent2_only": 8,
            "agent3_only": 3
        }
    }
    or None if venn diagram is disabled or not enough agents.
    """
    # Check if venn diagram is enabled
    if not benchmark.venn_diagram or not benchmark.venn_diagram.get("enabled", False):
        return None

    # Always use benchmark.primary_metric (no separate config)
    primary_metric = benchmark.primary_metric
    top_n = benchmark.venn_diagram.get("top_n_agents", 3)
    min_threshold = benchmark.venn_diagram.get("min_score_threshold", 0.0)

    if not primary_metric:
        return None

    # Determine if primary_metric is an expression
    is_expression = any(op in primary_metric for op in ['and(', 'or(', 'not('])

    # Sort agents by overall_weighted_score (which is computed from primary_metric)
    sorted_agents = sorted(
        agents_perf,
        key=lambda x: x.get("overall_weighted_score", 0) or 0,
        reverse=True,
    )

    # Filter by minimum threshold and take top N
    top_agents = [
        ap for ap in sorted_agents if (ap.get("overall_weighted_score", 0) or 0) >= min_threshold
    ][:top_n]

    if len(top_agents) < 2:
        # Need at least 2 agents for a meaningful Venn diagram
        return None

    agent_ids = [ap["agent"] for ap in top_agents]
    agent_sets = {}

    # For each top agent, load their comments file and extract diff_ids
    for agent in top_agents:
        agent_id = agent["agent"]
        # Find the comments file for this agent
        comments_files = list(eval_results_dir.glob(f"*{agent_id}*_comments.jsonl"))
        if not comments_files:
            return None

        # Use the most recent file (sorted by name gives chronological order)
        comments_file = sorted(comments_files)[-1]

        try:
            df = load_eval_comments(str(comments_file))
            if df.empty:
                agent_sets[agent_id] = []
                continue

            # Find diff_ids where primary_metric evaluates to True
            diff_ids_with_true = set()
            
            if is_expression:
                # Evaluate expression at comment level
                for diff_id in df["diff_id"].unique():
                    diff_rows = df[df["diff_id"] == diff_id]
                    try:
                        # Evaluate expression for each comment in this diff
                        evaluated = _evaluate_expression_on_df(primary_metric, diff_rows)
                        # If ANY comment in this diff evaluates to True → include diff_id
                        if evaluated.any():
                            diff_ids_with_true.add(str(diff_id))
                    except Exception:
                        pass
            else:
                # Simple column name (existing logic)
                if primary_metric not in df.columns:
                    agent_sets[agent_id] = []
                    continue
                
                for diff_id in df["diff_id"].unique():
                    diff_rows = df[df["diff_id"] == diff_id]
                    try:
                        has_true = bool(
                            diff_rows[primary_metric].astype("boolean").any(skipna=True)
                        )
                        if has_true:
                            diff_ids_with_true.add(str(diff_id))
                    except Exception:
                        pass

            agent_sets[agent_id] = sorted(list(diff_ids_with_true))
        except Exception as e:
            print(f"    [WARN] Error loading comments for {agent_id}: {e}")
            agent_sets[agent_id] = []

    # Add "Others" circle: all agents not in top N
    others_diffs = set()
    try:
        for comments_file in eval_results_dir.glob("*_comments.jsonl"):
            file_name = comments_file.name
            # Extract agent_id from filename
            # Try to match against top_agent ids
            is_top_agent = False
            for agent_id in agent_ids:
                if agent_id in file_name:
                    is_top_agent = True
                    break
            
            if not is_top_agent:
                # This is an "other" agent
                try:
                    df = load_eval_comments(str(comments_file))
                    if not df.empty:
                        if is_expression:
                            for diff_id in df["diff_id"].unique():
                                diff_rows = df[df["diff_id"] == diff_id]
                                evaluated = _evaluate_expression_on_df(primary_metric, diff_rows)
                                if evaluated.any():
                                    others_diffs.add(str(diff_id))
                        else:
                            if primary_metric in df.columns:
                                for diff_id in df["diff_id"].unique():
                                    diff_rows = df[df["diff_id"] == diff_id]
                                    has_true = bool(
                                        diff_rows[primary_metric].astype("boolean").any(skipna=True)
                                    )
                                    if has_true:
                                        others_diffs.add(str(diff_id))
                except Exception:
                    pass
    except Exception as e:
        print(f"    [WARN] Error loading 'Others' agents: {e}")
    
    # Add "Others" to agent_sets
    agent_sets["Others"] = sorted(list(others_diffs))

    # Compute intersections
    if len(agent_sets) < 2:
        return None

    agent_list = list(agent_sets.keys())
    set_list = [set(agent_sets[a]) for a in agent_list]

    intersections = {}

    # All agents together
    if len(agent_list) >= 3:
        intersections[f"all_{len(agent_list)}"] = len(set.intersection(*set_list))

    # Pairwise intersections
    for i in range(len(agent_list)):
        for j in range(i + 1, len(agent_list)):
            key = f"{agent_list[i]}_{agent_list[j]}"
            intersections[key] = len(set_list[i] & set_list[j])

    # Only in each set
    for i, agent in enumerate(agent_list):
        other_sets = set.union(*[set_list[j] for j in range(len(agent_list)) if j != i])
        only = len(set_list[i] - other_sets)
        intersections[f"{agent}_only"] = only

    # Total unique diff_ids across all agents
    all_diffs = set.union(*set_list)
    total_unique_diffs = len(all_diffs)

    # Debug output: Print venn diagram calculation details
    print(f"\n  === Venn Diagram Calculation ({benchmark.name}) ===")
    print(f"  Primary metric: {primary_metric}")
    for agent in agent_list:
        diffs = agent_sets[agent]
        print(f"    {agent:25} {len(diffs):3d} diffs detected")
    print(f"  Unique diffs detected (union): {total_unique_diffs}")
    if benchmark.dataset_total_diffs:
        uncovered = benchmark.dataset_total_diffs - total_unique_diffs
        uncovered_pct = (uncovered / benchmark.dataset_total_diffs * 100)
        print(f"  Total diffs in benchmark:     {benchmark.dataset_total_diffs}")
        print(f"  Uncovered diffs:              {uncovered} ({uncovered_pct:.1f}%)")

    return {
        "agents": agent_list,
        "sets": agent_sets,
        "intersections": intersections,
        "total_unique_diffs": total_unique_diffs,
    }


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

        # Extract submission metadata and evaluation version from first row
        # Read evaluation_version directly from JSONL to preserve string format
        model = ""
        timestamp = ""
        evaluation_version = None
        try:
            import json

            with open(comments_path, "r") as f:
                first_line = f.readline()
                if first_line:
                    first_obj = json.loads(first_line)
                    evaluation_version = first_obj.get("evaluation_version", None)
        except Exception:
            pass

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
        is_expression = any(op in primary_metric for op in ['and(', 'or(', 'not(']) if primary_metric else False
        
        if is_expression:
            # Evaluate expression across all comments and compute mean
            try:
                evaluated = _evaluate_expression_on_df(primary_metric, eval_df)
                overall_score = float(evaluated.mean()) if len(evaluated) > 0 else None
            except Exception as e:
                print(f"    [WARN] Error evaluating primary metric expression for {agent_id}: {e}")
                overall_score = None
        elif primary_metric and primary_metric in metric_results:
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
                "evaluation_version": evaluation_version,
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
    "task_accomplishment_rate": "Workflow Completion",
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
    "metric/human/is_llm_human_aligned": "LLM Alignment to Human",
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

        print(f"  Stems found: {len(stems)} submission(s)")

        agents_perf = _aggregate_benchmark(
            benchmark, stems, gt_df, benchmarks_root, dir_name=bname
        )

        # Compute Venn diagram data if enabled
        venn_diagram_data = None
        eval_results_dir = Path(benchmarks_root) / bname / "eval-results"
        if eval_results_dir.exists():
            venn_diagram_data = _compute_venn_diagram_data(
                benchmark, agents_perf, eval_results_dir, benchmarks_root
            )
            if venn_diagram_data:
                print(f"  Venn diagram: {len(venn_diagram_data['agents'])} agents")

        # Collect evaluation versions used
        eval_versions = set()
        for ap in agents_perf:
            all_agents.add(ap.get("agent", ""))
            all_models.add(ap.get("model", ""))
            total_diffs += ap.get("total_diffs", 0)
            total_comments += ap.get("total_comments", 0)
            if ap.get("evaluation_version"):
                eval_versions.add(ap["evaluation_version"])

        out_filename = f"data_{benchmark.name}.json"
        out_file = output_path / out_filename

        # Add venn diagram data to output if available
        output_data = agents_perf.copy()
        if venn_diagram_data:
            output_data = {"agents": agents_perf, "venn_diagram": venn_diagram_data}

        with open(out_file, "w") as f:
            json.dump(output_data, f, indent=2)

        # Show evaluation versioning summary
        version_summary = (
            ", ".join(str(v) for v in sorted(eval_versions)) if eval_versions else "N/A"
        )
        print(f"  Agents: {len(agents_perf)} | Eval Version(s): {version_summary}")
        print(f"  Written: {out_file}")
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

    # Generate evaluation versions HTML page
    _generate_evaluation_versions_page(benchmarks, benchmarks_root, Path(output_dir))

    print(f"\n=== Leaderboard updated ===")
    print(f"  Benchmarks : {len(output_filelist)}")
    print(f"  Agents     : {len(all_agents)}")
    print(f"  Models     : {len(all_models)}")
    print(f"  Diffs      : {total_diffs}")
    print(f"  Comments   : {total_comments}")


def _get_evaluator_description(evaluator_class: str) -> str:
    """Get human-readable description for an evaluator."""
    descriptions = {
        # Human alignment evaluators
        "human.IsLLMContextAligned": "Evaluates if LLM-generated comments are contextually appropriate and aligned with code context (LLM-based judge)",
        "human.IsLLMHumanAligned": "Assesses alignment between LLM comments and human expert reviews (LLM-based judge)",
        "human.IsHumanCommentLocationMatched": "Verifies if comment locations match human expert annotations",
        "human.IsHumanLLMLocationMatched": "Checks if LLM comment locations align with human expert locations",
        "human.LLMCommentBleuScore": "Measures lexical similarity between LLM and human comments using BLEU score",
        "human.LLMCommentEditSimilarityScore": "Calculates edit distance similarity between LLM and human comments",
        "human.LLMCommentRouge1Score": "Evaluates unigram overlap between LLM and human comments (ROUGE-1)",
        "human.LLMCommentRougeLScore": "Evaluates longest matching subsequence between LLM and human comments (ROUGE-L)",
        # Bug detection evaluators
        "bug.IsBugLocationMatched": "Verifies if identified bug locations match ground truth annotations",
        "bug.IsBugCommentRelevant": "Determines if a comment is relevant to the identified bug (LLM-based judge)",
        "bug.IsBugCommentTypeRelevant": "Checks if comment type matches the bug category",
        "bug.IsBugSuggestionValid": "Evaluates if bug fix suggestions are valid (LLM-based judge)",
        "bug.IsCommentLocationSuggestionMatched": "Matches suggested bug locations with ground truth",
        "bug.IsCommentLocationRelevantMatched": "Verifies relevance of suggested comment locations",
        "bug.IsCommentLocationRelevantMatchedRecall": "Measures recall of relevant comment locations",
        # Judge evaluators
        "judge.IsRelevantCommentDiff": "Evaluates if comment is relevant to the code diff being reviewed",
        "judge.IsCommentInformative": "Assesses if a comment provides useful information (LLM-based judge)",
        # Operational evaluators
        "ops.TrajectoryCostMetrics": "Calculates API costs based on token usage and LLM pricing",
    }
    return descriptions.get(evaluator_class, "No description available")


def _generate_evaluation_versions_page(
    benchmarks: list[str], benchmarks_root: str, output_path: Path
):
    """Generate evaluation-versions.html page from evaluation_versions.json files."""

    # Load base template
    template_file = (
        Path(__file__).parent.parent
        / "leaderboard"
        / "templates"
        / "evaluation-versions.html"
    )
    if not template_file.exists():
        print(f"  Warning: Template file not found: {template_file}")
        return

    with open(template_file) as f:
        template_content = f.read()

    # Build tab navigation and content for each benchmark
    tab_nav_parts = []
    tab_content_parts = []

    for idx, bname in enumerate(benchmarks):
        versions_file = Path(benchmarks_root) / bname / "evaluation_versions.json"
        changelog_file = Path(benchmarks_root) / bname / "evaluation_changelog.md"
        if not versions_file.exists():
            continue

        with open(versions_file) as f:
            versions_data = json.load(f)

        # Read changelog markdown if available
        changelog_content = ""
        if changelog_file.exists():
            with open(changelog_file) as f:
                changelog_content = f.read()

        benchmark = load_benchmark(bname, benchmarks_root=benchmarks_root)
        display_name = benchmark.display_name or bname
        tab_id = f"tab-{bname}"
        is_active = idx == 0

        # Build tab navigation item
        active_class = "active" if is_active else ""
        tab_nav_parts.append(
            f"""
                <li class="nav-item" role="presentation">
                    <button class="nav-link {active_class}" id="{tab_id}-tab" data-bs-toggle="tab" 
                            data-bs-target="#{tab_id}" type="button" role="tab">
                        {display_name}
                    </button>
                </li>"""
        )

        # Build tab content pane
        current_version = versions_data.get("current_version", "N/A")
        versions = versions_data.get("versions", {})

        version_cards = []
        for version, version_info in sorted(versions.items(), reverse=True):
            is_current = version == current_version
            badge = (
                '<span class="badge bg-success ms-2">Current</span>'
                if is_current
                else ""
            )

            released = version_info.get("released_date", "Unknown")
            evaluators = version_info.get("evaluators", [])

            # Extract description from changelog markdown if available
            description = version_info.get("changes", "No details provided")
            if changelog_content:
                import re

                # Find the version section in markdown (e.g., "## Version 1.0")
                pattern = rf"## Version {re.escape(version)}.*?\n\n\*\*Description:\*\* (.+?)\n"
                match = re.search(pattern, changelog_content, re.DOTALL)
                if match:
                    description = match.group(1).strip()

            evaluator_list = []
            for evaluator in evaluators:
                eval_class = evaluator.get("class", "Unknown")
                llm_model = evaluator.get("llm_model")
                if llm_model:
                    desc = _get_evaluator_description(eval_class)
                    evaluator_list.append(
                        f'<li><code>{eval_class}</code> <span class="badge bg-info">LLM: {llm_model}</span><br/><small class="text-muted">{desc}</small></li>'
                    )
                else:
                    desc = _get_evaluator_description(eval_class)
                    evaluator_list.append(
                        f'<li><code>{eval_class}</code><br/><small class="text-muted">{desc}</small></li>'
                    )

            version_cards.append(
                f"""
                <div class="card mb-3">
                    <div class="card-header">
                        <h6 class="mb-0 fw-bold">Version {version} {badge}</h6>
                    </div>
                    <div class="card-body">
                        <p class="text-muted small mb-3"><i class="fas fa-calendar me-2"></i>Released: {released}</p>
                        <div class="alert alert-info mb-3">
                            <strong>Description:</strong> {description}
                        </div>
                        <p class="mb-2"><strong>Evaluators ({len(evaluators)}):</strong></p>
                        <ul class="mb-0">
                            {''.join(evaluator_list)}
                        </ul>
                    </div>
                </div>"""
            )

        show_class = "show active" if is_active else ""
        tab_content_parts.append(
            f"""
            <div class="tab-pane fade {show_class}" id="{tab_id}" role="tabpanel">
                <div class="p-4">
                    {''.join(version_cards)}
                </div>
            </div>"""
        )

    tabs_nav_html = "\n".join(tab_nav_parts)
    tabs_content_html = "\n".join(tab_content_parts)

    # Replace placeholders in template
    html_content = template_content.replace("{{TAB_NAVIGATION}}", tabs_nav_html)
    html_content = html_content.replace("{{TAB_CONTENT}}", tabs_content_html)

    output_file = output_path / "evaluation-versions.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    print(f"  Written: {output_file}")


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
