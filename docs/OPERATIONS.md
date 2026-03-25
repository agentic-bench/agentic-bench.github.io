# Operations Guide

## Dependency Sync

```bash
uv sync
```

## Two-Step Pipeline

### Step 1 — Evaluate

```bash
cd src

# Select LLM gateway (generic or direct), necessary configuration e.g., .env required
export LLM_GATEWAY_BACKEND=generic

# Evaluate on AgenticCR-Verified
uv run src/evaluator.py \
    --benchmark agenticcr-verified \
    --input ../benchmarks/agenticcr-verified/llm-comments/my-agent_20260301-0900.jsonl \
    --benchmarks-root ../benchmarks

# Evaluate on SCRBench
uv run src/evaluator.py \
    --benchmark scrbench \
    --input ../benchmarks/scrbench/llm-comments/my-agent_20260301-0900.jsonl \
    --benchmarks-root ../benchmarks
```

**Options:**
- `--benchmark` — benchmark directory name (e.g. `agenticcr-verified`, `scrbench`)
- `--input` — path to llm-comments JSONL file
- `--benchmarks-root` — root folder containing benchmark directories (default: `../benchmarks`)
- `--resume` — skip already-evaluated rows (default: overwrite)

The evaluator:
1. Loads `benchmark_info.json` to discover which evaluator classes to run
2. Reads the submission's `submission` block to identify the agent
3. Runs each evaluator class in order (supports composite metrics that depend on prior results)
4. Writes `{stem}_comments.jsonl` and `{stem}_trajectory.jsonl` to `eval-results/`
5. Logs all LLM judge calls to `{stem}_eval.log`

### Step 2 — Generate leaderboard

```bash
cd src

uv run src/leaderboard.py \
    --benchmarks-root ../benchmarks \
    --output-dir ../leaderboard
```

**Options:**
- `--benchmarks-root` — root folder with benchmark directories (default: `../benchmarks`)
- `--output-dir` — where to write `data/*.json` (default: `../leaderboard`)

The leaderboard:
1. Discovers all benchmarks in order of `tab_order` from `benchmark_info.json`
2. For each benchmark, discovers all eval-result stems (latest per `agent_id` wins)
3. Aggregates metrics using the mode declared in `benchmark_info.json → metric_aggregation`
4. Computes per-group summary scores (`group_summary`)
5. Computes `task_accomplishment_rate` using `task_accomplishment_mode`
6. Writes `data_*.json`, `benchmark_meta.json`, `metric_display_names.json`, `output_filelist.json`, `statistics.json`

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

---

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

The leaderboard's `primary_metric` can be an expression combining multiple metrics:

```json
"primary_metric": "and(metric/human/is_llm_human_aligned, metric/human/is_human_llm_location_matched)"
```

**Supported operators:**
- `and(A, B)` — both must be true (used for composite requirements)
- `or(A, B)` — either can be true
- `not(A)` — inverse of metric

**How it's used:**
1. The evaluator computes `and()` as a boolean per-comment
2. The leaderboard aggregates using the mode declared in `metric_aggregation` (or default `precision` for bool)
3. The leaderboard then computes the **harmonic mean** of the underlying metrics for ranking (if configured in `group_summary`)
4. Venn diagrams automatically detect and use the expression to build their sets

**Example:** For AgenticCR-Verified with `and(location, alignment)`:
- Evaluator: computes boolean `True` only if comment is both localized AND aligned
- Leaderboard: aggregates as `mean(bool)` = % of comments that are both localized AND aligned
- Display: Shows "Code Review Capability" as harmonic mean of location + alignment metrics
- Venn: Builds circles of diffs with at least one comment matching the `and()` expression

---

## `benchmark_info.json` Reference

```json
{
  "name":                    "agenticcr-verified",
  "display_name":            "AgenticCR-Verified",
  "benchmark_goal":          "human-alignment",
  "tab_order":               1,
  "dataset_total_diffs":     362,
  "task_accomplishment_mode": "submitted",
  "is_ground_truth_eligible": true,

  "evaluator_classes": [
    "human.IsHumanLLMLocationMatched",
    "human.LLMCommentRouge1Score",
    "human.LLMCommentRougeLScore",
    "human.LLMCommentBleuScore",
    "human.LLMCommentEditSimilarityScore",
    "human.IsLLMHumanAligned",
    "human.IsLLMContextAligned"
  ],

  "metric_aggregation": {
    "metric/human/is_human_llm_location_matched":     "recall",
    "metric/human/llm_comment_rouge1_score":          "mean",
    "metric/human/llm_comment_rougel_score":          "mean",
    "metric/human/llm_comment_bleu_score":            "mean",
    "metric/human/llm_comment_edit_similarity_score": "mean",
    "metric/human/is_llm_human_aligned":              "precision",
    "metric/human/is_llm_context_aligned":            "precision"
  },

  "leaderboard": {
    "primary_metric": "and(metric/human/is_llm_human_aligned, metric/human/is_human_llm_location_matched)",
    // Supports expressions: and(), or(), not() for composite metrics
    // Simple metric names still work for backward compatibility

    "column_groups": {
      "Code Review Capability": [...metric keys...],
      "Text Similarity":        [...metric keys...],
      "Trajectory":             [...trajectory keys...]
    },

    "group_summary": {
      "Code Review Capability": {
        "method":  "harmonic_mean",
        "columns": [...metric keys to average...]
      },
      "Text Similarity": {
        "method":  "mean",
        "columns": [...metric keys to average...]
      },
      "Trajectory": {
        "method": "pick",
        "column": "trajectory/total_costs"
      }
    },

    "display_names": {
      "metric/human/is_llm_human_aligned": "Human Alignment"
    }
  },

  "venn_diagram": {
    "enabled": true,
    "top_n_agents": 3,
    "min_score_threshold": 0.0
    // NOTE: No "primary_metric" field - always uses leaderboard.primary_metric
    // Supports expression-based metrics automatically
  }
}
```

**Key fields:**
- `tab_order` — integer controlling benchmark tab order in leaderboard (lower = leftmost)
- `dataset_total_diffs` — total number of diffs in the full benchmark dataset (used for task accomplishment rate)
- `task_accomplishment_mode` — `"submitted"` (presence in llm-comments = accomplished) or `"has_reviews"` (must have ≥1 non-empty review)
- `group_summary.method` — `"mean"` averages all listed columns; `"harmonic_mean"` computes harmonic mean; `"pick"` reads a single column directly
- `evaluator_classes` — ordered list; composite metrics must come after their dependencies
- `venn_diagram.enabled` — set to `true` to enable venn diagram visualization
- `venn_diagram.top_n_agents` — number of top agents to include (default: 3)
- `venn_diagram.min_score_threshold` — minimum score to be eligible (default: 0.0)
- **Venn diagram always uses `leaderboard.primary_metric`** — no separate configuration needed, supports expressions automatically

---

## How to Submit

1. **Generate comments** using `benchmarks/{benchmark}/input-dataset/dataset.jsonl`
2. **Format** your output as the unified llm-comments JSONL (see [Data Formats](#data-formats) or `leaderboard/format.html`)
3. **Save** as `benchmarks/{benchmark}/llm-comments/{agent_id}_{YYYYMMDD-HHMM}.jsonl`
4. **Evaluate** (Step 1 above) — produces `eval-results/{stem}_comments.jsonl` and `{stem}_trajectory.jsonl`
5. **Update leaderboard** (Step 2 above) — regenerates all `data/*.json` files
6. **Open a PR** with:
   - `benchmarks/{benchmark}/llm-comments/{stem}.jsonl`
   - `benchmarks/{benchmark}/eval-results/{stem}_comments.jsonl`
   - `benchmarks/{benchmark}/eval-results/{stem}_trajectory.jsonl`
   - Updated `leaderboard/data/*.json`

---

## Evaluator Architecture

The evaluator system is built on the `BaseEvaluator` class in `src/evaluators/base.py`:

```python
class BaseEvaluator:
    evaluation_name = "metric/group/evaluator_name"
    
    def evaluate(self, diff_row, submission, prior_evals) -> dict:
        """Returns {metric_name: score} or {metric_name: None} if not applicable."""
        pass
```

**Key evaluators:**

| Class | Module | Purpose | Output |
|---|---|---|---|
| `TrajectoryCostMetrics` | `ops/` | Calculates monetary costs from token counts | `trajectory_input_costs`, `trajectory_output_costs`, `trajectory_total_costs` |
| `IsHumanLLMLocationMatched` | `human/` | Location matching with fuzzy window | `metric/human/is_human_llm_location_matched` |
| `IsLLMHumanAligned` | `human/` | Overall human alignment (composite) | `metric/human/is_llm_human_aligned` |
| `IsBugLocationMatched` | `bug/` | Exact location match for security bugs | `metric/bug/is_bug_location_matched` |
| `IsBugCommentTypeRelevant` | `bug/` | CWE-aware semantic matching | `metric/bug/is_bug_comment_type_relevant` |
| ~~`IsLLMContextAligned`~~ | ~~`judge/`~~ | ~~LLM-as-judge: context relevance~~ | ~~Removed - unused~~ |
| ~~`IsCommentInformative`~~ | ~~`judge/`~~ | ~~LLM-as-judge: informativeness~~ | ~~Removed - unused~~ |
| ~~`IsRelevantCommentDiff`~~ | ~~`judge/`~~ | ~~LLM-as-judge: relevance~~ | ~~Removed - unused~~ |

**Evaluation order matters:**
- Evaluators listed in `benchmark_info.json → evaluator_classes` run in order
- Composite metrics (e.g., `IsLLMHumanAligned`) depend on prior evaluators
- Results from prior evaluators are passed via `prior_evals` parameter

---

## Evaluation Versioning

Each benchmark maintains an **evaluation version** that represents a snapshot of all metrics and LLM judges used for that benchmark. This allows tracking when evaluation criteria change and helps maintain result comparability.

### Philosophy

- **Version represents the entire evaluator set** - Not individual evaluators
- **Independent per benchmark** - AgenticCR-Verified and SCRBench have separate version numbers
- **Semantic versioning** - MAJOR.MINOR format
  - **MAJOR bump** (e.g., 1.0 → 2.0): Add/remove evaluator, change LLM model, major prompt changes
  - **MINOR bump** (e.g., 1.0 → 1.1): Model version update, bug fixes, minor prompt tweaks

### Files

**Per-benchmark versioning files:**

- `benchmarks/{benchmark}/evaluation_versions.json` - Machine-readable version definitions
  ```json
  {
    "current_version": "1.0",
    "versions": {
      "1.0": {
        "released_date": "2026-01-10",
        "evaluators": [
          {"class": "human.IsLLMHumanAligned", "llm_model": "gpt-5.1-2025-11-13"},
          {"class": "ops.TrajectoryCostMetrics", "llm_model": null}
        ]
      }
    }
  }
  ```

- `benchmarks/{benchmark}/evaluation_changelog.md` - Human-readable changelog with version descriptions

### Workflow for Maintainers

#### When to Bump Versions

**MAJOR version bump:**
- Adding a new evaluator
- Removing an evaluator
- Switching to a different LLM model (e.g., gpt-4 → gpt-5)
- Major changes to evaluation prompts or logic

**MINOR version bump:**
- Updating to a newer version of the same model (e.g., gpt-5.0 → gpt-5.1)
- Bug fixes in evaluator code
- Minor prompt improvements

#### How to Bump a Version

1. **Update evaluation_versions.json** - Change `current_version` and add new version entry
2. **Update evaluation_changelog.md** - Add description of changes
3. **Update benchmark_info.json** - Change `current_version` field
4. **Run leaderboard generator** - `python src/leaderboard.py` to regenerate with new version

### Viewing Evaluation Versions

- **In leaderboard table:** Hover over agent name to see evaluation version in tooltip
- **Evaluation Versions page:** Click "Evaluation Versions" button in navbar for full version history

⚠️ **Important:** Results from different evaluation versions may not be directly comparable.

---

## leaderboard HTML

The leaderboard is a self-contained static HTML app in `leaderboard/`.

| Page | URL | Description |
|---|---|---|
| Main leaderboard | `index.html` | Two-tab table with sorting, group collapse, Pareto plot |
| Submission format | `format.html` | Full schema documentation for llm-comments format |
| AgenticCR-Verified | `benchmark-agenticcr-verified.html` | Examples, groundtruth, metrics |
| SCRBench | `benchmark-scrbench.html` | Examples, groundtruth, metrics, CWE info |

**Features:**
- Light / dark theme toggle (persisted in `localStorage`)
- Metric group columns collapse/expand — collapsed state shows one representative summary score per group
- Pareto frontier scatter plot (cost vs. overall score) — `[Experimental]`
- All configuration driven by `benchmark_meta.json` — no JS changes needed for new metrics or benchmarks

To deploy, copy `leaderboard/` to any static file host or serve locally:
```bash
cd leaderboard && python3 -m http.server 8080
```
