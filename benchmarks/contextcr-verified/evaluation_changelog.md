# ContextCR-Verified Evaluation Changelog

## Version 1.0 (Released 2026-01-10)

**Description:** Initial evaluation version - baseline metric set for human alignment

### Evaluators (8 total)

**LLM-based (2):**
- `human.IsLLMContextAligned` - Uses `gpt-5.1-2025-11-13`
- `human.IsLLMHumanAligned` - Uses `gpt-5.1-2025-11-13`

**Non-LLM (6):**
- `human.IsHumanCommentLocationMatched`
- `human.LLMCommentBleuScore`
- `human.LLMCommentEditSimilarityScore`
- `human.LLMCommentRouge1Score`
- `human.LLMCommentRougeLScore`
- `ops.TrajectoryCostMetrics`

### Version Bumping Guidelines

**→ MAJOR version (e.g., 1.0 → 2.0) when:**
- Adding or removing an evaluator
- Changing LLM model (e.g., gpt-4 → gpt-5)
- Major prompt change in an LLM evaluator

**→ MINOR version (e.g., 1.0 → 1.1) when:**
- Updating LLM model version (e.g., gpt-5.0 → gpt-5.1)
- Bug fixes in evaluators
- Minor prompt improvements
- Non-breaking changes to metric calculation

### Leaderboard Configuration
- **Primary metric (v1.0+):** Signal-to-Noise Ratio (SNR)
  - Formula: `SNR = A / (T - A)` where A = aligned comments, T = total comments
  - Penalizes high-volume generators with few aligned comments
  - Encourages efficiency: more signal (aligned) per noise (misaligned + non-localized)
  - Stored as `overall_weighted_score` (hidden from leaderboard view)

- **Displayed primary metric:** Code Review Capability (harmonic mean)
  - Harmonic mean of `is_llm_human_aligned` and `is_human_llm_location_matched`
  - Formula: `2 / (1/alignment + 1/localization)`
  - Penalizes imbalance: both metrics must be reasonably high for good score
  - Encourages balanced performance across alignment and localization
  - Stored as `group_score/Code Review Capability` (visible as fixed column)

- **Reference metrics (hidden from view, kept in data):**
  - `overall_weighted_score`: SNR for efficiency analysis
  - `conditional_alignment`: % aligned among localized comments
  - `is_llm_context_aligned`: Context alignment (evaluated but not displayed)
  - `is_llm_hunk_context_aligned`: Hunk-level context alignment (evaluated but not displayed)

### Notes
- All evaluators are from `src/evaluators/` directory
- See `evaluation_versions.json` for detailed version snapshot
- Leaderboard supports expression-based primary metrics (and/or/not operations)
