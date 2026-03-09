# ContextCRBench Evaluation Changelog

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

### Notes
- All evaluators are from `src/evaluators/` directory
- See `evaluation_versions.json` for detailed version snapshot
