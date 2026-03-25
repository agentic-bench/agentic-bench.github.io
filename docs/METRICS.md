# Metrics Reference

## Metric Aggregation Modes

Declared in `benchmark_info.json → metric_aggregation` (key: metric name, value: mode string).

| Mode | Formula | When to use |
|---|---|---|
| `precision` | `mean(bool per comment)` | Binary per-comment metrics (default for `bool` returns) |
| `mean` | `mean(float per comment)` | Continuous scores (default for `float` returns, e.g. ROUGE) |
| `recall` | `covered_gt / total_gt` (capped per diff) | Metrics measuring ground-truth coverage |
| `sum_per_diff` | `mean(per-diff sums)` | Trajectory metrics summed within a diff (not yet used; trajectory uses `_trajectory.jsonl` directly) |

Metrics not listed in `metric_aggregation` default to `precision` for `bool` returns and `mean` for `float` returns.

---

## Expression-Based Primary Metrics

The `primary_metric` field in `benchmark_info.json → leaderboard` supports **expression syntax** to create composite metrics from existing evaluators:

### Supported Operations

- **`and(metric1, metric2)`** — Both conditions must be true (logical AND)
- **`or(metric1, metric2)`** — Either condition is true (logical OR)  
- **`not(metric)`** — Negation of a metric

### Evaluation Flow

1. **Expression is evaluated at comment-level** — Each comment gets a boolean result
2. **Aggregation to diff-level** — For venn diagrams, a diff is included if ANY comment evaluates to true
3. **Overall score** — Mean of boolean results across all comments (precision)

### Example

```json
{
  "leaderboard": {
    "primary_metric": "and(metric/human/is_llm_human_aligned, metric/human/is_human_llm_location_matched)"
  }
}
```

This composite metric requires comments to be **both semantically aligned with human reviewers AND correctly localized** to the right file/line.

**Benefits:**
- No hardcoding — expressions live entirely in config
- Metrics are still evaluated independently (all data preserved)
- Backward compatible — simple metric names still work
- Venn diagram automatically uses the same expression

**Note:** Individual metrics in the expression must still be declared in `evaluator_classes` and will appear in eval-results files. Use `column_groups` to control leaderboard visibility.

---

## Trajectory & Cost Metrics

The pipeline uses a dedicated `TrajectoryCostMetrics` evaluator to calculate cost metrics:

- **Input:** Token counts from `submission.trajectory` (`input_tokens`, `output_tokens`) and `submission.model` name
- **Process:** Uses the `genai-prices` library to map model names to provider pricing and compute costs
- **Output:** `trajectory_input_costs`, `trajectory_output_costs`, `trajectory_total_costs` in `*_trajectory.jsonl`

**Data separation:**
- `llm-comments/*.jsonl` (submissions): Contains only token counts and steps, **NO costs**
- `eval-results/*_comments.jsonl`: Comment-level metrics only, **NO trajectory field**
- `eval-results/*_trajectory.jsonl`: Full trajectory data including computed costs

This design ensures:
1. Costs are never submitted by agents — they're always computed deterministically
2. Clear distinction between input data (tokens) and computed outputs (costs)
3. Pareto frontier plot (cost vs. performance) is always consistent and reproducible
