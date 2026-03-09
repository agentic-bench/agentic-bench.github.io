# evaluator.py
#
# Description: Run benchmark evaluation on a unified llm-comments submission file.
#
# Usage:
#   cd src
#   python evaluator.py --benchmark <benchmark_name> \
#       --input <path/to/llm-comments.jsonl> \
#       [--benchmarks-root ../benchmarks]
#
# The submission file must conform to the unified llm-comments format (see README.md).
#
# Outputs (written to benchmarks/<benchmark_name>/eval-results/):
#   {agent_id}_{timestamp}_comments.jsonl    – one row per comment with metric columns
#   {agent_id}_{timestamp}_trajectory.jsonl  – one row per diff_id with trajectory fields
#
# Re-running is safe: already-evaluated metric columns are skipped.

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

import evaluators as evaluators_pkg
from dataloader import (
    BenchmarkRegistry,
    SubmissionMeta,
    load_benchmark,
    load_benchmark_ground_truth,
    load_benchmark_dataset,
    load_llm_comments,
    load_eval_comments,
    parse_submission_filename,
)
from evaluators.base import BaseEvaluator
import json

logger = logging.getLogger(__name__)

SAVE_INTERVAL = 50  # flush to disk every N rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_evaluator_classes(benchmark: BenchmarkRegistry) -> List[type]:
    """Resolve evaluator class objects from names listed in benchmark_info.json.

    The evaluators __init__.py registers classes under dotted keys like
    'human.IsHumanLLMLocationMatched' and 'bug.IsBugLocationMatched'.
    We try that dotted key first (direct getattr), then fall back to a
    walk through subpackage attributes.
    """
    classes = []
    for name in benchmark.evaluator_classes:
        # 1. Try the dotted key directly (how __init__.py registers them)
        obj = getattr(evaluators_pkg, name, None)

        # 2. Fallback: walk attribute chain (future-proof)
        if obj is None:
            parts = name.split(".")
            obj = evaluators_pkg
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break

        if obj is not None and inspect.isclass(obj):
            classes.append(obj)
        else:
            logger.warning(f"Evaluator class not found: {name}")
    return classes


def _get_ground_truth_for_diff(diff_id: str, gt_df: pd.DataFrame | None) -> list[dict]:
    if gt_df is None or gt_df.empty:
        return []
    return gt_df[gt_df["diff_id"] == diff_id].to_dict(orient="records")


def _get_diff_text(diff_id: str, dataset_df: pd.DataFrame | None) -> str:
    """Look up diff text from the benchmark dataset (for judge evaluators)."""
    if dataset_df is None or dataset_df.empty or "diff" not in dataset_df.columns:
        return ""
    matched = dataset_df[dataset_df["diff_id"] == diff_id]
    return str(matched.iloc[0].get("diff", "")) if not matched.empty else ""


def _load_evaluation_metadata(benchmark_name: str, benchmarks_root: str) -> dict:
    """Load evaluation version from evaluation_versions.json.
    
    Returns dict with:
      - evaluation_version: version string to store in eval-results
      - _full_metadata: full version info for console display only (not stored)
    """
    versions_path = Path(benchmarks_root) / benchmark_name / "evaluation_versions.json"
    
    if not versions_path.exists():
        logger.warning(f"evaluation_versions.json not found: {versions_path}")
        return {}
    
    try:
        with open(versions_path) as f:
            versions_data = json.load(f)
        
        current_version = versions_data.get("current_version")
        if not current_version:
            logger.warning(f"No current_version in {versions_path}")
            return {}
        
        version_info = versions_data.get("versions", {}).get(current_version, {})
        
        # Build minimal metadata for storage (only version number)
        metadata = {
            "evaluation_version": current_version
        }
        
        # Build full metadata for console display (not stored in files)
        llm_models = []
        for evaluator in version_info.get("evaluators", []):
            if evaluator.get("llm_model"):
                llm_models.append({
                    "evaluator": evaluator["class"],
                    "model": evaluator["llm_model"]
                })
        
        metadata["_full_metadata"] = {
            "evaluation_date": version_info.get("released_date", ""),
            "llm_models_used": llm_models
        }
        
        return metadata
    except Exception as ex:
        logger.error(f"Error loading evaluation metadata: {ex}")
        return {}


def _infer_metric_dtype(col: str) -> str:
    """Infer pandas dtype from metric column name."""
    if "_score" in col:
        return "float64"
    return "boolean"


def _save_comments(df: pd.DataFrame, path: str, eval_metadata: dict = None):
    # Drop trajectory field before saving — it's only used internally
    # to build the separate _trajectory.jsonl file
    df_to_save = df.drop(columns=["trajectory"], errors="ignore")
    
    # Add only evaluation_version to each row (minimal metadata)
    if eval_metadata and "evaluation_version" in eval_metadata:
        df_to_save["evaluation_version"] = eval_metadata["evaluation_version"]
    
    df_to_save.to_json(path, orient="records", lines=True)


def _save_trajectory(traj_df: pd.DataFrame, path: str, eval_metadata: dict = None):
    # Add only evaluation_version to each row (minimal metadata)
    if eval_metadata and "evaluation_version" in eval_metadata:
        traj_df["evaluation_version"] = eval_metadata["evaluation_version"]
    
    traj_df.to_json(path, orient="records", lines=True)


def _build_trajectory_df(comments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract one trajectory row per diff_id from the comments dataframe.
    The trajectory dict is identical for all comments in the same diff,
    so we take the first occurrence per diff_id.

    Also merges trajectory-level metrics computed by trajectory evaluators.
    """
    rows = []
    for diff_id, group in comments_df.groupby("diff_id", sort=False):
        traj = group.iloc[0].get("trajectory", {})
        if isinstance(traj, dict):
            row = {"diff_id": diff_id, **traj}
        else:
            row = {"diff_id": diff_id}

        # Merge trajectory metrics if available
        if hasattr(comments_df, "_trajectory_metrics"):
            metrics = comments_df._trajectory_metrics.get(diff_id, {})
            if isinstance(metrics, dict):
                row.update(metrics)

        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------


def run_evaluator(
    benchmark_name: str,
    input_path: str,
    benchmarks_root: str = "benchmarks",
):
    # 1. Load benchmark config and evaluation metadata
    benchmark = load_benchmark(benchmark_name, benchmarks_root=benchmarks_root)
    evaluator_classes = _get_evaluator_classes(benchmark)
    eval_metadata = _load_evaluation_metadata(benchmark_name, benchmarks_root)

    if not evaluator_classes:
        raise ValueError(
            f"No evaluator classes resolved for benchmark '{benchmark_name}'."
        )
    
    # Log evaluation version info
    if eval_metadata.get("evaluation_version"):
        print(f"\n=== Evaluation Version: {eval_metadata['evaluation_version']} ===")
        full_meta = eval_metadata.get("_full_metadata", {})
        if full_meta.get("llm_models_used"):
            print("  LLM Models Used:")
            for model_info in full_meta["llm_models_used"]:
                print(f"    - {model_info['evaluator']}: {model_info['model']}")
    else:
        logger.warning("No evaluation version metadata found")

    # 2. Parse submission metadata from filename
    filename = Path(input_path).name
    parsed = parse_submission_filename(filename)
    agent_id = parsed["agent_id"]
    timestamp = parsed["timestamp"]
    stem = f"{agent_id}_{timestamp}" if timestamp else agent_id

    # 3. Load ground truth + dataset (for diff text)
    gt_df = load_benchmark_ground_truth(benchmark_name, benchmarks_root=benchmarks_root)
    dataset_df = load_benchmark_dataset(benchmark_name, benchmarks_root=benchmarks_root)

    # 4. Load llm-comments → flat per-comment DataFrame
    submission, comments_df = load_llm_comments(input_path)

    if comments_df.empty:
        raise ValueError(f"No comments found in '{input_path}'.")

    # Use agent_id from submission block if present, else from filename
    agent_id = submission.agent_id or agent_id
    timestamp = submission.timestamp or timestamp
    model = submission.model

    print(f"\n=== Evaluating: {agent_id} on {benchmark_name} ===")
    print(f"  Input   : {input_path}")
    print(f"  Diffs   : {comments_df['diff_id'].nunique()}")
    print(f"  Comments: {len(comments_df)}")

    # 5. Determine output paths
    eval_dir = Path(benchmarks_root) / benchmark_name / "eval-results"
    eval_dir.mkdir(parents=True, exist_ok=True)

    comments_path = str(eval_dir / f"{stem}_comments.jsonl")
    trajectory_path = str(eval_dir / f"{stem}_trajectory.jsonl")

    # 6. Resume from existing comments file if present
    if Path(comments_path).exists():
        existing_df = load_eval_comments(comments_path)
        # Merge existing metric columns back into comments_df
        metric_cols = [c for c in existing_df.columns if c.startswith("metric/")]
        for col in metric_cols:
            comments_df[col] = (
                existing_df[col].values
                if len(existing_df) == len(comments_df)
                else pd.NA
            )
        # Check if existing evaluation version matches current
        if eval_metadata:
            existing_version = existing_df.iloc[0].get("evaluation_version") if not existing_df.empty else None
            current_version = eval_metadata.get("evaluation_version")
            if existing_version and existing_version != current_version:
                logger.warning(
                    f"Evaluation version mismatch: "
                    f"existing={existing_version}, current={current_version}. "
                    f"Previous results will be overwritten."
                )
        print(f"  Resuming from existing: {comments_path}")
        print(f"  Existing metric columns: {metric_cols}")

    # 7. Set log file env var for BaseEvaluator loggers
    # Ensure log directory exists
    eval_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = eval_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str((logs_dir / f"{stem}_eval.log").resolve())
    os.environ[BaseEvaluator.LOG_ENV_VAR] = log_path
    logger.info(f"LLM judge call log → {log_path}")

    # 8. Run each evaluator column
    for evaluator_cls in evaluator_classes:
        col = evaluator_cls.evaluation_name

        # Check if this is a trajectory-level evaluator (returns dict with multiple metrics)
        is_trajectory_evaluator = col.startswith("metric/ops/trajectory_")

        if is_trajectory_evaluator:
            # For trajectory evaluators, we process once per diff_id
            print(f"\n--- Trajectory Evaluator: {col} ---")

            # Track which diffs we've already processed
            processed_diffs = set()
            trajectory_results = {}

            for i, (idx, row) in enumerate(
                tqdm(comments_df.iterrows(), total=len(comments_df))
            ):
                diff_id = row.get("diff_id")

                # Skip if we already processed this diff
                if diff_id in processed_diffs:
                    continue

                processed_diffs.add(diff_id)

                try:
                    row_dict = row.to_dict()
                    # Inject diff text for judge/context evaluators
                    row_dict["_diff"] = _get_diff_text(diff_id, dataset_df)
                    # Inject ground truth list
                    gt = _get_ground_truth_for_diff(diff_id, gt_df)
                    result = evaluator_cls.evaluate(row_dict, gt)

                    # Store trajectory results (result is a dict with multiple metric keys)
                    if result is not None and isinstance(result, dict):
                        trajectory_results[diff_id] = result

                except Exception as ex:
                    logger.error(
                        f"Evaluation error — evaluator={col}, "
                        f"diff_id={diff_id}: {ex}"
                    )
                    trajectory_results[diff_id] = {}

            # Apply trajectory results to trajectory file (will be created later)
            # Store in a temporary dict on comments_df for later extraction
            if not hasattr(comments_df, "_trajectory_metrics"):
                comments_df._trajectory_metrics = {}
            comments_df._trajectory_metrics.update(trajectory_results)

        else:
            # Original per-comment evaluator logic
            dtype = _infer_metric_dtype(col)

            # Initialize column if missing
            if col not in comments_df.columns:
                comments_df[col] = pd.NA

            # Cast to correct dtype (ignore failures — mixed NAs handled below)
            try:
                comments_df[col] = comments_df[col].astype(dtype)
            except Exception:
                pass

            print(f"\n--- Evaluator: {col} ({dtype}) ---")

            for i, (idx, row) in enumerate(
                tqdm(comments_df.iterrows(), total=len(comments_df))
            ):
                # Skip already-evaluated rows
                val = comments_df.at[idx, col]
                try:
                    if pd.notna(val):
                        continue
                except Exception:
                    pass

                try:
                    row_dict = row.to_dict()
                    # Inject diff text for judge/context evaluators
                    row_dict["_diff"] = _get_diff_text(
                        row_dict.get("diff_id", ""), dataset_df
                    )
                    # Inject ground truth list
                    gt = _get_ground_truth_for_diff(row_dict.get("diff_id", ""), gt_df)
                    result = evaluator_cls.evaluate(row_dict, gt)
                except Exception as ex:
                    logger.error(
                        f"Evaluation error — evaluator={col}, "
                        f"diff_id={row.get('diff_id')}: {ex}"
                    )
                    result = None

                comments_df.at[idx, col] = result

                if i > 0 and i % SAVE_INTERVAL == 0:
                    _save_comments(comments_df, comments_path, eval_metadata)

            _save_comments(comments_df, comments_path, eval_metadata)

    # 9. Final save of comments
    _save_comments(comments_df, comments_path, eval_metadata)

    # 10. Write trajectory file (one row per diff_id — no redundancy)
    traj_df = _build_trajectory_df(comments_df)
    _save_trajectory(traj_df, trajectory_path, eval_metadata)

    # 11. Summary
    print(f"\n=== Evaluation Summary ===")
    print(f"  Benchmark   : {benchmark_name}")
    print(f"  Agent       : {agent_id}  (model: {model})")
    if eval_metadata.get("evaluation_version"):
        full_meta = eval_metadata.get("_full_metadata", {})
        eval_date = full_meta.get("evaluation_date", "N/A")
        print(f"  Eval Version: {eval_metadata['evaluation_version']} ({eval_date})")
        if full_meta.get("llm_models_used"):
            llm_count = len(full_meta["llm_models_used"])
            print(f"  LLM Judges  : {llm_count} evaluator(s)")
    print(f"  Diffs       : {comments_df['diff_id'].nunique()}")
    print(f"  Comments    : {len(comments_df)}")
    print(f"  Comments →  : {comments_path}")
    print(f"  Trajectory →: {trajectory_path}")

    metric_cols = [c for c in comments_df.columns if c.startswith("metric/")]
    if metric_cols:
        agg_cfg = benchmark.metric_aggregation
        print("\n  Metric summary (per-comment aggregation):")
        for col in metric_cols:
            agg = agg_cfg.get(col, "mean" if "_score" in col else "precision")
            try:
                if "_score" in col:
                    val = comments_df[col].astype(float).mean(skipna=True)
                    print(f"    {col}: mean={val:.4f}")
                else:
                    val = comments_df[col].astype("boolean").mean(skipna=True)
                    print(f"    {col}: rate={val:.4f}  [agg={agg}]")
            except Exception:
                print(f"    {col}: (could not summarize)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a unified llm-comments submission against a benchmark."
    )
    parser.add_argument(
        "--benchmark",
        "-b",
        required=True,
        help="Benchmark name (must have a benchmark_info.json in benchmarks/)",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        dest="input_path",
        help="Path to the llm-comments JSONL submission file",
    )
    parser.add_argument(
        "--benchmarks-root",
        default="benchmarks",
        help="Root folder for benchmarks (default: benchmarks)",
    )
    args = parser.parse_args()

    run_evaluator(
        benchmark_name=args.benchmark,
        input_path=args.input_path,
        benchmarks_root=args.benchmarks_root,
    )
