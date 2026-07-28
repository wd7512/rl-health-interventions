"""Convert sprint1 transition tables from old format to new format.

Old format (per persona directory):
  - day_boundary.json: pipe-delimited key → sleep transition probs
  - within_day_N.json (N=0..4): pipe-delimited key → step_bin transition probs

New format (per persona directory):
  - step_0.json: global_state + transitions array (sleep + step_bin combined)
  - step_N.json (N=0..4): per-step transition files

Usage:
    uv run python -m scripts.convert_sprint1_tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sprint1 config constants ──────────────────────────────────────────────────
_STOCHASTIC_FACTORS = ("step_bin", "sleep")
_ACTIONS = ("idle", "movement_suggestion", "goal_reminder", "journal")
_STEP_COUNT = 5

# Old format field order for pipe-delimited keys
_WITHIN_DAY_FIELDS = ("step_bin", "burden", "action", "day_of_week", "sleep")
_BOUNDARY_FIELDS = ("step_bin", "burden", "day_of_week", "sleep")

_PROB_EPSILON = 1e-10

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TABLES_ROOT = _REPO_ROOT / "tables" / "persona"


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _parse_pipe_key(key: str, fields: tuple[str, ...]) -> dict[str, str]:
    """Parse a pipe-delimited key into a dict of field → value."""
    parts = key.split("|")
    if len(parts) != len(fields):
        msg = (
            f"Expected {len(fields)} parts in key {key!r} "
            f"(fields={fields}), got {len(parts)}"
        )
        raise ValueError(msg)
    return dict(zip(fields, parts, strict=True))


def _identity_probs(current_value: str, domain: tuple[str, ...]) -> dict[str, float]:
    """Build an identity distribution (current value stays with prob 1.0)."""
    return {v: 1.0 if v == current_value else 0.0 for v in domain}


# ── Loading helpers ───────────────────────────────────────────────────────────


def _load_boundary(
    path: Path,
) -> dict[tuple[str, str, str, str], dict[str, float]]:
    """Load day_boundary.json, return state_key → sleep probs map."""
    raw_data: Any = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for key, value in raw_data.items():
        parsed = _parse_pipe_key(key, _BOUNDARY_FIELDS)
        state_key = (
            parsed["step_bin"],
            parsed["burden"],
            parsed["day_of_week"],
            parsed["sleep"],
        )
        result[state_key] = {k: float(v) for k, v in value["_"].items()}
    return result


def _load_within_day(
    path: Path,
) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    """Load within_day_N.json, return state_key → step_bin probs map."""
    raw_data: Any = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for key, value in raw_data.items():
        parsed = _parse_pipe_key(key, _WITHIN_DAY_FIELDS)
        state_key = (
            parsed["step_bin"],
            parsed["burden"],
            parsed["action"],
            parsed["day_of_week"],
            parsed["sleep"],
        )
        result[state_key] = {k: float(v) for k, v in value["_"].items()}
    return result


def _validate_probs(probs: dict[str, float], label: str) -> None:
    """Validate that a probability distribution sums to 1.0."""
    total = sum(probs.values())
    if abs(total - 1.0) > _PROB_EPSILON:
        logger.warning(
            "Probabilities for %s sum to %s (expected 1.0): %s",
            label,
            total,
            probs,
        )


# ── Entry building ────────────────────────────────────────────────────────────


def _make_entry(
    step_bin: str,
    burden: str,
    dow: str,
    sleep: str,
    action: str,
    step_bin_probs: dict[str, float],
    sleep_probs: dict[str, float],
) -> dict:
    """Build a single transition entry dict for the new format."""
    return {
        "state": {
            "step_bin": step_bin,
            "sleep": sleep,
            "day_of_week": dow,
            "burden": burden,
        },
        "action": action,
        "next_state_probs": {
            "step_bin": step_bin_probs,
            "sleep": sleep_probs,
        },
    }


def _build_step_entries(
    boundary_data: dict[tuple[str, str, str, str], dict[str, float]],
    within_data: dict[tuple[str, str, str, str, str], dict[str, float]],
) -> list[dict]:
    """Build transition entries for step 0 by merging boundary and within_day_0."""
    entries: list[dict] = []

    for (step_bin, burden, action, dow, sleep), step_bin_probs in within_data.items():
        state_key = (step_bin, burden, dow, sleep)

        # Sleep probs from boundary data (same for all actions at given state)
        sleep_probs = boundary_data.get(state_key)
        if sleep_probs is None:
            logger.warning(
                "No boundary entry for state %s; using sleep identity",
                state_key,
            )
            sleep_probs = _identity_probs(sleep, ("good", "poor"))

        _validate_probs(step_bin_probs, f"step_0 step_bin {state_key} action={action}")
        _validate_probs(sleep_probs, f"step_0 sleep {state_key} action={action}")

        entries.append(
            _make_entry(
                step_bin, burden, dow, sleep, action, step_bin_probs, sleep_probs
            )
        )

    return entries


def _build_step_entries_identity(
    within_data: dict[tuple[str, str, str, str, str], dict[str, float]],
) -> list[dict]:
    """Build transition entries for steps 1..4 (step_bin from within, id sleep)."""
    entries: list[dict] = []

    for (step_bin, burden, action, dow, sleep), step_bin_probs in within_data.items():
        sleep_probs = _identity_probs(sleep, ("good", "poor"))

        _validate_probs(
            step_bin_probs,
            f"within_day step_bin {step_bin}/{burden}/{dow}/{sleep}",
        )

        entries.append(
            _make_entry(
                step_bin, burden, dow, sleep, action, step_bin_probs, sleep_probs
            )
        )

    return entries


def _write_step_file(
    path: Path,
    *,
    step_of_day: int,
    entries: list[dict],
) -> None:
    """Write a step file in the new format."""
    data = {
        "global_state": {"step_of_day": step_of_day},
        "transitions": entries,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s (%d entries)", path, len(entries))


# ── Conversion ────────────────────────────────────────────────────────────────


def _convert_persona(persona_dir: Path) -> list[Path]:
    """Convert one persona directory from old format to new format.

    Returns the list of newly created step file paths.
    """
    persona_name = persona_dir.name
    logger.info("Converting persona: %s", persona_name)

    # Load old-format files
    boundary_path = persona_dir / "day_boundary.json"
    within_paths = [persona_dir / f"within_day_{i}.json" for i in range(_STEP_COUNT)]

    if not boundary_path.exists():
        logger.warning("Skipping %s: missing day_boundary.json", persona_name)
        return []

    boundary_data = _load_boundary(boundary_path)
    within_data: list[dict[tuple[str, str, str, str, str], dict[str, float]]] = []
    for p in within_paths:
        if not p.exists():
            logger.warning("Skipping %s: missing %s", persona_name, p.name)
            return []
        within_data.append(_load_within_day(p))

    created_files: list[Path] = []

    # Step 0: merge boundary (sleep) + within_day_0 (step_bin)
    step_0 = _build_step_entries(boundary_data, within_data[0])
    step_0_path = persona_dir / "step_0.json"
    _write_step_file(step_0_path, step_of_day=0, entries=step_0)
    created_files.append(step_0_path)

    # Steps 1..4: within_day_N (step_bin) + sleep identity
    for step_idx in range(1, _STEP_COUNT):
        entries = _build_step_entries_identity(within_data[step_idx])
        step_path = persona_dir / f"step_{step_idx}.json"
        _write_step_file(step_path, step_of_day=step_idx, entries=entries)
        created_files.append(step_path)

    return created_files


# ── Validation ────────────────────────────────────────────────────────────────


def _check_next_state_probs_factor(
    nsp: dict[str, Any], factor: str, prefix: str
) -> list[str]:
    """Check a single stochastic factor within next_state_probs."""
    if factor not in nsp:
        return [f"{prefix}: next_state_probs missing '{factor}'"]
    probs = nsp[factor]
    if not isinstance(probs, dict):
        return [f"{prefix}: next_state_probs['{factor}'] is not a dict"]
    total = sum(probs.values())
    if abs(total - 1.0) > _PROB_EPSILON:
        return [f"{prefix}: next_state_probs['{factor}'] sums to {total}, expected 1.0"]
    return []


def _check_entry_next_state_probs(nsp: Any, prefix: str) -> list[str]:
    """Check next_state_probs for a single entry."""
    if not isinstance(nsp, dict):
        return [f"{prefix}: 'next_state_probs' is not a dict"]
    errors: list[str] = []
    for factor in _STOCHASTIC_FACTORS:
        errors.extend(_check_next_state_probs_factor(nsp, factor, prefix))
    return errors


def _check_entry_state(state: Any, prefix: str) -> list[str]:
    """Check required state keys in a transition entry."""
    if not isinstance(state, dict):
        return [f"{prefix}: 'state' is not a dict"]
    errors: list[str] = []
    for key in ("step_bin", "sleep", "day_of_week", "burden"):
        if key not in state:
            errors.append(f"{prefix}: state missing '{key}'")
    return errors


def _check_entry_action(action_val: Any, prefix: str) -> list[str]:
    """Check action in a transition entry."""
    if not isinstance(action_val, str):
        return [f"{prefix}: 'action' is not a string"]
    if action_val not in _ACTIONS:
        return [f"{prefix}: unknown action '{action_val}'"]
    return []


def _validate_entry(entry: Any, filename: str, index: int) -> list[str]:
    """Validate a single transition entry."""
    prefix = f"{filename}[{index}]"

    if not isinstance(entry, dict):
        return [f"{prefix}: transition entry is not a dict"]

    if "state" not in entry:
        return [f"{prefix}: missing 'state'"]

    errors: list[str] = []
    errors.extend(_check_entry_state(entry["state"], prefix))

    if "action" not in entry:
        errors.append(f"{prefix}: missing 'action'")
    else:
        errors.extend(_check_entry_action(entry["action"], prefix))

    if "next_state_probs" not in entry:
        errors.append(f"{prefix}: missing 'next_state_probs'")
        return errors

    errors.extend(_check_entry_next_state_probs(entry["next_state_probs"], prefix))
    return errors


def _validate_step_file(path: Path, step_idx: int) -> list[str]:
    """Validate a single step file."""
    errors: list[str] = []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"Invalid JSON in {path.name}: {e}"]

    gs = data.get("global_state", {})
    if gs.get("step_of_day") != step_idx:
        errors.append(
            f"{path.name}: global_state.step_of_day expected {step_idx}, "
            f"got {gs.get('step_of_day')}"
        )

    transitions = data.get("transitions", [])
    if not isinstance(transitions, list):
        errors.append(f"{path.name}: transitions is not a list")
        return errors

    for i, entry in enumerate(transitions):
        errors.extend(_validate_entry(entry, path.name, i))

    return errors


def _validate_persona(persona_dir: Path) -> list[str]:
    """Validate a converted persona directory.

    Returns a list of validation error messages (empty if valid).
    """
    errors: list[str] = []
    for step_idx in range(_STEP_COUNT):
        path = persona_dir / f"step_{step_idx}.json"
        if not path.exists():
            errors.append(f"Missing {path.name}")
        else:
            errors.extend(_validate_step_file(path, step_idx))
    return errors


def _remove_old_files(persona_dir: Path) -> None:
    """Remove old-format files after successful conversion."""
    old_files = [
        persona_dir / "day_boundary.json",
        *(persona_dir / f"within_day_{i}.json" for i in range(_STEP_COUNT)),
    ]
    for path in old_files:
        if path.exists():
            path.unlink()
            logger.info("Removed old file: %s", path.name)


# ── Main ──────────────────────────────────────────────────────────────────────


def _process_persona(persona_dir: Path) -> bool:
    """Convert and validate one persona directory. Returns True on success."""
    created = _convert_persona(persona_dir)
    if not created:
        logger.warning("No files created for %s, skipping", persona_dir.name)
        return False

    validation_errors = _validate_persona(persona_dir)
    if validation_errors:
        logger.error(
            "Validation failed for %s:\n  %s",
            persona_dir.name,
            "\n  ".join(validation_errors),
        )
        return False

    _remove_old_files(persona_dir)
    logger.info("Converted %s: %d step files", persona_dir.name, len(created))
    return True


def _ensure_tables_root() -> list[Path]:
    """Validate tables root and return sorted persona directories."""
    if not _TABLES_ROOT.is_dir():
        logger.error("Tables root not found: %s", _TABLES_ROOT)
        raise SystemExit(1)

    persona_dirs = sorted(d for d in _TABLES_ROOT.iterdir() if d.is_dir())
    if not persona_dirs:
        logger.warning("No persona directories found in %s", _TABLES_ROOT)
    return persona_dirs


def _convert_all(persona_dirs: list[Path]) -> tuple[int, int]:
    """Convert all persona directories. Returns (success_count, error_count)."""
    success_count = 0
    error_count = 0

    for persona_dir in persona_dirs:
        logger.info("─" * 50)
        try:
            if _process_persona(persona_dir):
                success_count += 1
            else:
                error_count += 1
        except Exception:
            logger.exception("Error converting %s", persona_dir.name)
            error_count += 1

    return success_count, error_count


def main() -> None:
    """Find and convert all persona directories."""
    persona_dirs = _ensure_tables_root()
    if not persona_dirs:
        return

    success_count, error_count = _convert_all(persona_dirs)

    logger.info("=" * 50)
    logger.info(
        "Conversion complete: %d succeeded, %d failed",
        success_count,
        error_count,
    )
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
