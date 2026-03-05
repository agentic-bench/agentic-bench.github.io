#!/bin/bash
set -euo pipefail  # Exit on any error, undefined variables, or pipe failures

# Configuration from template
REPO_URL="https://github.com/microsoft/terminal"
COMMIT_HASH="f3d0ec2e467c582c33592047eca0bb2c1c7f1ad4"

echo "Setting up ContextCRBench environment..."
echo "Repository: $REPO_URL"
echo "Commit: $COMMIT_HASH"

# Clone the repository
echo "Cloning repository..."
if ! git clone "$REPO_URL.git" repo; then
    echo "ERROR: Failed to clone repository $REPO_URL"
    exit 1
fi

cd repo

# Validate that we have a valid git repository with commits
echo "Validating repository state..."
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "ERROR: Repository has no commits yet"
    echo "Repository state:"
    git status || true
    exit 1
fi
echo "Repository validation passed - has commits"

# Fetch and checkout the specific commit
echo "Fetching commit: $COMMIT_HASH"
if ! git fetch origin "$COMMIT_HASH" && ! git fetch origin --unshallow; then
    echo "ERROR: Failed to fetch commit $COMMIT_HASH"
    exit 1
fi

echo "Checking out commit $COMMIT_HASH..."
if ! git checkout "$COMMIT_HASH"; then
    echo "ERROR: Failed to checkout commit $COMMIT_HASH"
    echo "Available commits:"
    git log --oneline -10 || echo "No commits found"
    exit 1
fi

echo "Successfully checked out commit: $(git rev-parse --short HEAD)"

# Validate that checkout worked and we're at the right commit
CURRENT_COMMIT=$(git rev-parse HEAD)
if [ "$CURRENT_COMMIT" != "$COMMIT_HASH" ]; then
    echo "ERROR: Checkout validation failed"
    echo "Expected: $COMMIT_HASH"
    echo "Actual: $CURRENT_COMMIT"
    exit 1
fi
echo "Checkout validation passed"

# Revert the commit into staged changes for code review
echo "Setting up code review environment..."
echo "Reverting commit to show changes as staged modifications..."

# Validate that we have a parent commit to reset to
if ! git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    echo "ERROR: No parent commit found - cannot reset HEAD~1"
    echo "This might be the initial commit in the repository"
    git log --oneline || echo "No commit history available"
    exit 1
fi

if ! git reset HEAD~1; then
    echo "ERROR: Failed to reset to previous commit"
    exit 1
fi
echo "Successfully reset to previous commit"

# Validate that reset worked properly and we're not in a "No commits yet" state
if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "ERROR: After reset, repository shows 'No commits yet'"
    echo "Git status:"
    git status || true
    exit 1
fi
echo "Reset validation passed - repository has valid state"


# Stage all changes (including new files) for review
echo "Staging all changes for review..."
if ! git add -N .; then
    echo "ERROR: Failed to stage intent-to-add files"
    exit 1
fi
echo "Successfully staged all changes for review"

# Context file is already available in workspace for agent access
echo "Context file available at: /workspace/code_change_context.json"
if [ ! -f "/workspace/code_change_context.json" ]; then
    echo "ERROR: Context file not found at /workspace/code_change_context.json"
    exit 1
fi

# Final validation before completing setup
echo "Performing final environment validation..."

# Ensure git status doesn't show "No commits yet"
if git status 2>&1 | grep -q "No commits yet"; then
    echo "ERROR: Final validation failed - git status shows 'No commits yet'"
    echo "Full git status output:"
    git status || true
    echo "Git log output:"
    git log --oneline -5 || echo "No commits in log"
    exit 1
fi

# Ensure we have staged changes ready for review
STAGED_COUNT=$(git status --porcelain | wc -l)
if [ "$STAGED_COUNT" -eq 0 ]; then
    echo "ERROR: No staged changes found for code review"
    echo "Git status:"
    git status
    echo "This might indicate the reset/staging process failed"
    exit 1
fi

# Display final environment status
echo ""
echo "=== ContextCRBench Environment Ready ==="
echo "Repository: $REPO_URL"
echo "Base commit: $(git rev-parse --short HEAD) ($(git log -1 --format='%s'))"
echo "Working directory: /workspace/repo"
echo "Staged changes: $STAGED_COUNT files"
echo ""
echo "📋 Code Review Status:"
git status --short | head -10
if [ $(git status --porcelain | wc -l) -gt 10 ]; then
    echo "   ... and $(($(git status --porcelain | wc -l) - 10)) more files"
fi
echo ""
echo "✅ All validations passed - ready for code review!"

# Return to workspace root - agents will work from here
cd /workspace

# Setup complete - the CMD from Dockerfile will start the shell
echo "Setup complete! Starting interactive shell in /workspace..."