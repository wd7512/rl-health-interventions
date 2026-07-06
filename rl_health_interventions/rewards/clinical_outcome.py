import numpy as np
from typing import List, Optional

class ClinicalOutcomeGenerator:
    """Generates synthetic clinical outcomes based on daily steps over a 3-week period."""

    def __init__(self, baseline_hba1c: float = 7.0, baseline_sbp: float = 130.0):
        self.baseline_hba1c = baseline_hba1c
        self.baseline_sbp = baseline_sbp

    def compute_hba1c_change(self, daily_steps: List[float]) -> float:
        """
        Synthetic HbA1c change: linear improvement with increased step average relative to baseline.
        HbA1c reduction is capped at -1.0% (maximum improvement).
        """
        if len(daily_steps) < 21:
            return 0.0  # not enough data
        # Use last 21 days
        recent_steps = daily_steps[-21:]
        avg_steps = np.mean(recent_steps)
        # Assume each 1000 steps above 5000 reduces HbA1c by 0.1%
        target_steps = 7000.0
        if avg_steps < 3000:
            change = 0.2  # worsening
        elif avg_steps < target_steps:
            change = -0.05 * (avg_steps - 3000) / 1000
        else:
            change = -0.1 * (avg_steps - target_steps) / 1000 - 0.2
        change = max(-1.0, min(0.5, change))  # clamp
        return change

    def compute_sbp_change(self, daily_steps: List[float]) -> float:
        """
        Synthetic systolic blood pressure change: linear improvement with step average.
        Maximum reduction capped at -10 mmHg.
        """
        if len(daily_steps) < 21:
            return 0.0
        recent_steps = daily_steps[-21:]
        avg_steps = np.mean(recent_steps)
        # Assume each 1000 steps above 5000 reduces SBP by 2 mmHg
        target_steps = 7000.0
        if avg_steps < 3000:
            change = 5.0  # increase
        elif avg_steps < target_steps:
            change = -1.0 * (avg_steps - 3000) / 1000
        else:
            change = -2.0 * (avg_steps - target_steps) / 1000 - 4.0
        change = max(-10.0, min(10.0, change))
        return change

    def compute_clinical_score(self, daily_steps: List[float]) -> float:
        """
        Composite clinical score between -1 and 1.
        Positive indicates improvement.
        """
        hba1c_change = self.compute_hba1c_change(daily_steps)
        sbp_change = self.compute_sbp_change(daily_steps)
        # Normalize and combine: HbA1c change of -1% is max improvement, SBP -10 mmHg is max
        norm_hba1c = -hba1c_change  # negative change is improvement -> positive score
        norm_sbp = -sbp_change / 10.0
        score = 0.5 * norm_hba1c + 0.5 * norm_sbp
        score = max(-1.0, min(1.0, score))
        return score
