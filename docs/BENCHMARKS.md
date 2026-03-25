# Benchmarks

## ContextCR-Verified (`contextcr-verified`)
- **Goal:** human-alignment — does the agent identify the same issues as human expert reviewers?
- **Dataset:** 362 real GitHub pull requests across multiple open-source projects
- **Ground truth:** 561 human expert review comments (file, line, content)
- **Primary metric:** `and(metric/human/is_llm_human_aligned, metric/human/is_human_llm_location_matched)` — composite metric requiring BOTH human alignment AND correct localization
- **Task accomplishment mode:** `submitted` — any diff present in llm-comments counts as accomplished
- **Comprehensive dataset:** Full issue/PR context and commit diffs in `commit_diff/` directory

### Dataset Annotation & Quality Control

The dataset was manually reviewed and annotated to remove low-quality ground truth. Starting from **421 initial pull requests with 680 ground truth comments**, a verification process identified and excluded problematic annotations:

- **59 fully excluded tasks** (all annotations invalid) — removed entirely from dataset.jsonl
- **18 partially excluded tasks** (some annotations invalid) — retained in dataset.jsonl with only valid ground truth lines
- **119 excluded ground truth lines** — removed from specific (file, line) positions within partially excluded tasks
- **Final cleaned dataset:** 362 tasks with 561 valid ground truth comments

**Exclusion reasons:**
- **OoC (Out of Context) ground truth** (26 tasks) — Annotations refer to code not present in the diff
- **Vague ground truth** (14 tasks) — Ambiguous or unclear annotations that don't provide meaningful feedback
- **Change too large** (13 tasks) — Pull request scope too broad, annotations not specific enough
- **Bot comments** (3 tasks) — Automated tool-generated comments lacking semantic value
- **All changes removed** (3 tasks) — Diff content changed making original annotations invalid

**Example excluded tasks:**
- `airflow_issue_42331_pr_42277_xl_fac840e2` — change too large
- `aspnetcore_issue_28335_pr_28763_l_3b5d4b24` — vague ground truth  
- `gitea_issue_31002_pr_31003_sm_e67258d8` — OoC ground truth
- `aseprite_issue_4781_pr_4925_l_52393980` — bot comment
- `osu_issue_14015_pr_14017_sm_749d7a7b` — all changes removed

---

## SCRBench (`scrbench`)
- **Goal:** bug-capacity — can the agent identify real security bugs (CVEs/CWEs)?
- **Dataset:** 144 pull requests containing known security vulnerabilities
- **Ground truth:** 243 bug locations with CWE ID, description, and patch validity
- **Primary metric:** `metric/bug/is_comment_location_relevant_matched`
- **Task accomplishment mode:** `has_reviews` — the agent must produce at least one review comment
- **Vulnerability typing:** New `metric/bug/is_bug_comment_type_relevant` uses CWE grouping for semantic matching
