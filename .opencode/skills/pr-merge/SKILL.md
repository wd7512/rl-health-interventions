---
name: pr-merge
description: Use when a GitHub PR has an open review with unresolved comment threads that block merge, or when merging any PR with review commentary.
---

# PR Merge

Use this skill when a PR is blocked by unresolved review comments and needs
to be merged. Covers the full pipeline: resolve threads, verify CI, merge, clean up.

## Prerequisites

- `gh` CLI authenticated with repo scope
- You are the PR author or have write access
- All suggested changes have been addressed in the code (either committed or
  intentionally deferred with rationale)

## Workflow

### 1. Identify blocking state

```powershell
gh pr view <N> --json mergeStateStatus,reviewDecision,reviews,comments
```

- `mergeStateStatus: "BLOCKED"` — something is blocking merge
- `reviewDecision: ""` — no blocking review (COMMENTED is fine)
- Check if `mergeStateStatus` becomes `"CLEAN"` after resolving threads

### 2. Get review threads and their node IDs

```powershell
$query = '{"query": "query { repository(owner: \"<owner>\", name: \"<repo>\") { pullRequest(number: <N>) { reviewThreads(first: 20) { nodes { id isResolved comments(first: 5) { nodes { id body } } } } } } }"}'
Set-Content -Path "$env:TEMP\gh_threads.json" -Value $query -NoNewline -Encoding ascii
gh api graphql --input "$env:TEMP\gh_threads.json"
```

Each unresolved thread has a `PullRequestReviewThread` node ID (e.g. `PRRT_kw...`)
under `id`. Threads missing a reply from the PR author should get one.

### 3. Reply to each thread (optional but good practice)

```powershell
gh api repos/<owner>/<repo>/pulls/<N>/comments -f body="<reply>" -F in_reply_to=<comment_id>
```

Note: `-f` sends strings, `-F` sends JSON-typed values. `in_reply_to` must be
numeric, so use `-F`.

### 4. Resolve each thread via GraphQL

```powershell
$body = @'
{"query": "mutation { resolveReviewThread(input: {threadId: \"<THREAD_ID>\"}) { thread { id isResolved } } }"}
'@
Set-Content -Path "$env:TEMP\gh_mutation.json" -Value $body -NoNewline -Encoding ascii
gh api graphql --input "$env:TEMP\gh_mutation.json"
```

Repeat for each unresolved thread. Verify: `"isResolved": true`.

### 5. Confirm merge state is clean

```powershell
gh pr view <N> --json mergeStateStatus
# Expect: "CLEAN"
```

### 6. Merge

```powershell
gh pr merge <N> --squash --subject "<commit title>" --body "<commit body>"
```

### 7. Verify merge

```powershell
gh pr view <N> --json state,mergedAt,mergedBy
# Expect: "MERGED"
```

### 8. Clean up local branch

```powershell
git checkout main
git pull origin main
git branch -D <feature-branch>
```

## Notes

- PowerShell has no heredoc (`<<'EOF'`). Use `Set-Content` + temp files for
  multiline GraphQL payloads.
- Squash merging is preferred for feature branches with multiple WIP commits.
- The `resolveReviewThread` mutation requires the `PullRequestReviewThread` node
  ID, NOT the comment node ID (they have different prefixes: `PRRT_` vs `PRRC_`).
- Regular (non-review) PR comments do not block merging — only review comment
  threads do.
