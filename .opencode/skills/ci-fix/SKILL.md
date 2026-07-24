---
description: "Use when CI checks fail, a PR build is broken, thread-gate fails, lint errors appear, or the user asks how to fix CI. Maps each check name to exact fix commands."
mode: subagent
model: opencode-go/deepseek-v4-flash
permission:
  read: allow
  edit: deny
  bash: deny
---

# CI Fix

Look up the failing check name. Run the fix commands. Re-run the check.

## thread-gate

**Fails with:** `N review thread(s) are unresolved. Reply to each thread and resolve it before merging.`

**Fix:**
1. List unresolved threads: `gh api graphql -f query='query($owner:String!,$repo:String!,$pr:Int!){repository(owner:$owner,name:$repo){pullRequest(number:$pr){reviewThreads(first:20){nodes{id,isResolved,comments(first:1){nodes{author{login},body}}}}}}}' -F owner=OWNER -F repo=REPO -F pr=PR_NUM --jq '.data.repository.pullRequest.reviewThreads.nodes[]|select(.isResolved==false)|{id,body:.comments.nodes[0].body[:120]}'`
2. For each thread, post a reply: `gh api graphql -f query='mutation($tid:ID!,$b:String!){addPullRequestReviewThreadReply(input:{pullRequestReviewThreadId:$tid,body:$b}){comment{id}}}' -F tid=THREAD_ID -F b="YOUR_REPLY"`
3. Resolve each thread: `gh api graphql -f query='mutation($tid:ID!){resolveReviewThread(input:{threadId:$tid}){thread{id,isResolved}}}' -F tid=THREAD_ID`

**Fails with:** `N review thread(s) were resolved without a comment.`

**Fix:** Same as above — post a reply explaining the fix, then resolve. Even auto-resolved threads (marked by bots) need a human reply.

## lint

**Fails with:** ruff errors (format, unused imports, line length, etc.)

**Fix:**
1. Auto-fix: `uv run ruff check --fix .`
2. Format: `uv run ruff format .`
3. Verify: `uv run ruff format --check . && uv run ruff check`

## build

**Fails with:** import errors, missing dependencies, build system issues

**Fix:**
1. Sync dependencies: `uv sync --all-extras`
2. Check pyproject.toml for missing deps
3. Verify: `uv build`

## test-unit / test-integration

**Fails with:** test failures

**Fix:**
1. Run tests locally: `uv run pytest`
2. Read the failure output carefully
3. Fix the code or update the test
4. Re-run: `uv run pytest`

## check-linked-issue

**Fails with:** PR description doesn't reference an issue

**Fix:** Add one of these to your PR description:
- `Closes #123`
- `Fixes #123`
- `Resolves #123`

Or ensure the PR body contains a link to the issue.

## type-check

**Fails with:** ty type errors

**Fix:**
1. Run type check: `uv run ty check --exclude tests/`
2. Fix type annotations
3. Re-run: `uv run ty check --exclude tests/`
