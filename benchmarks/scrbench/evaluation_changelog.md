# SCRBench Evaluation Changelog

## Version 1.0 (Released 2026-01-10)

**Description:** Initial evaluation version - baseline metric set for bug capacity

### Evaluators (8 total)

**LLM-based (2):**
- `bug.IsBugCommentRelevant` - Uses `gpt-4.1-2025-04-14`
- `judge.IsBugSuggestionValid` - Uses `gpt-4.1-2025-04-14`

**Non-LLM (6):**
- `bug.IsBugLocationMatched`
- `bug.IsBugCommentTypeRelevant`
- `bug.IsCommentLocationRelevantMatched`
- `bug.IsCommentLocationSuggestionMatched`
- `bug.IsCommentLocationRelevantMatchedRecall`
- `ops.TrajectoryCostMetrics`

### Version Bumping Guidelines

**→ MAJOR version (e.g., 1.0 → 2.0) when:**
- Adding or removing an evaluator
- Changing LLM model (e.g., gpt-4 → gpt-5)
- Major prompt change in an LLM evaluator

**→ MINOR version (e.g., 1.0 → 1.1) when:**
- Updating LLM model version (e.g., gpt-4.0 → gpt-4.1)
- Bug fixes in evaluators
- Minor prompt improvements
- Non-breaking changes to metric calculation

### Notes
- All evaluators are from `src/evaluators/` directory
- See `evaluation_versions.json` for detailed version snapshot
