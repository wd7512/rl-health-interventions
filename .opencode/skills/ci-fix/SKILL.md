---
name: ci-fix
description: Fix CI failures for common checks — thread-gate, check-linked-issue, lint, build, test, type-check.
---

# CI Fix

Maps failing CI check names to exact fix procedures.

## thread-gate

**Failure:** `Check review thread resolution` — review threads resolved without replies.

**Fix:**
1. Reply to each unresolved thread explaining the fix
2. Then resolve the thread

```bash
# List unresolved threads
gh api graphql -f query='
query {
  repository(owner: "OWNER", name: "REPO") {
    pullRequest(number: PR_NUMBER) {
      reviewThreads(first: 10) {
        nodes {
          id
          isResolved
          comments(first: 1) {
            nodes {
              databaseId
              body
            }
          }
        }
      }
    }
  }
}' --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | {threadId: .id, commentId: .comments.nodes[0].databaseId, body: .comments.nodes[0].body[0:100]}'

# Reply to each thread
gh api repos/OWNER/REPO/pulls/comments/COMMENT_ID/replies -X POST -f body="Fixed in commit SHA. Brief explanation of the fix."

# Then resolve the thread
gh api graphql -f query='mutation { resolveReviewThread(input: {threadId: "THREAD_ID"}) { thread { isResolved } } }'
```

**Rule:** Never resolve a thread without replying first. The CI check enforces this.

## check-linked-issue

**Failure:** PR must link an issue using `Closes #N`, `Fixes #N`, or `Resolves #N`, AND the linked issue must not have the `needs-refinement` label.

**Fix:**
1. Add linked issue to PR body if missing:
```bash
gh pr edit PR_NUMBER --body "..."  # ensure body contains "Closes #N"
```

2. Remove `needs-refinement` label from linked issue:
```bash
gh issue edit ISSUE_NUMBER --remove-label "needs-refinement"
```

**Rule:** Before linking an issue to a PR, ensure it doesn't have the `needs-refinement` label. The CI check enforces this.

## lint

**Failure:** `ruff check` or `ruff format` failed.

**Fix:**
```bash
uv run ruff format .
uv run ruff check --fix
```

## build

**Failure:** `uv build` failed.

**Fix:** Check for packaging errors, missing dependencies, or invalid pyproject.toml. Run `uv build` locally to see the error.

## test-unit / test-integration

**Failure:** pytest failed.

**Fix:** Run `uv run pytest` locally, fix failing tests, ensure all pass before pushing.

## type-check

**Failure:** `ty check` failed.

**Fix:** Run `uv run ty check --exclude tests/` locally, fix type errors.

## Workflow

1. Identify failing check from `gh pr view PR_NUMBER --json statusCheckRollup`
2. Apply the fix procedure above
3. Push changes
4. Wait for CI to re-run
5. Verify all checks pass
