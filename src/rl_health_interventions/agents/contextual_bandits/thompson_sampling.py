from __future__ import annotations

# Re-export Posterior from the main thompson_sampling module so that tests
# can import it from either path.
from rl_health_interventions.agents.thompson_sampling import Posterior  # noqa: F401