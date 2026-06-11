Fix the swarm review findings in PR #73 to get it ready to merge. Address these in priority order:

CRITICAL FIXES:

1. WISDM timestamp unit - The timestamps in WISDM are in nanoseconds, not milliseconds. Find the from_epoch call in load_wisdm and change time_unit="ms" to time_unit="ns". Verify by checking the actual data: the timestamps are like 49105962326000 which is ~49105 seconds in nanoseconds (about 13.6 hours), which makes sense for a day of activity data.

2. Remove unused dependencies from pyproject.toml - Remove kaggle (only kagglehub is used), matplotlib, and seaborn from the dependencies list. These are not imported anywhere in the codebase.

HIGH FIXES:

3. Remove dead code - Find and remove the _cache_path function that is defined but never called anywhere in loaders.py.

4. Fix MHEALTH cache marker - The .mhealth_done check logs "already cached" but does not return early. Either add an early return that reads cached data, or remove the marker check entirely (since MHEALTH uses ucimlrepo which does not support local caching anyway).

5. Fix WESAD docstring - The docstring promises wrist columns but the code only extracts chest. Update the docstring to accurately describe what is extracted (chest signals only).

SECURITY HARDENING (Medium priority):

6. Add path traversal protection to tar/zip extraction in load_wisdm, load_extrasensory, and load_4tu_step_goals. For tar: check that resolved member path is within target dir. For zip: validate paths manually.

7. Reorder ExtraSensory URLs to try HTTPS first, remove the HTTP URL.

TESTS (add to tests/unit/data/test_loaders.py):

8. Create a new test file tests/unit/data/test_loaders.py with:
- Test _check_kaggle_auth with monkeypatch (tmp_path for file, env vars)
- Test _check_physionet_auth with monkeypatch
- Test load_synthetic returns a polars DataFrame with correct shape
- Test load_all returns a dict with all 12 dataset names as keys
- Test that each loader returns None when credentials are missing (mock the auth check functions)
- Test WISDM semicolon parsing with a small fixture

Keep tests simple and fast - use monkeypatch/mocks, not real network calls. Run uv run ruff check src tests and uv run pytest tests/ -q after.

After all fixes: git add -A, git commit with message describing changes, git push.
