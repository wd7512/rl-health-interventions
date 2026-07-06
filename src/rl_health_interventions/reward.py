"""Reward functions for the health intervention environment.

Includes daily step reward and 3-week clinical outcome reward.
"""

from typing import Optional
from .clinical_reward import ClinicalOutcomeSimulator, compute_clinical_reward


class HealthReward:
    """Combines daily step reward with periodic clinical improvement reward.

    The clinical reward is computed every 21 days and added to the total
    episode return. Daily step reward provides immediate feedback.
    """

    def __init__(
        self,
        step_target: float = 7500.0,
        step_reward_scale: float = 0.01,
        bp_scale: float = -0.5,
        hba1c_scale: float = -1.0,
        clinical_frequency: int = 21
    ):
        self.step_target = step_target
        self.step_reward_scale = step_reward_scale
        self.bp_scale = bp_scale
        self.hba1c_scale = hba1c_scale
        self.clinical_frequency = clinical_frequency
        self.clinical_sim = ClinicalOutcomeSimulator()
        self.day_count = 0
        self.episode_clinical_reward = 0.0

    def reset(self) -> None:
        """Reset reward state for new episode."""
        self.clinical_sim.reset()
        self.day_count = 0
        self.episode_clinical_reward = 0.0

    def compute_daily_step_reward(self, steps: float) -> float:
        """Reward for daily step count (positive for meeting target)."""
        diff = steps - self.step_target
        # Quadratic penalty below target, linear bonus above
        if diff < 0:
            reward = -self.step_reward_scale * (diff ** 2)
        else:
            reward = self.step_reward_scale * diff
        return reward

    def __call__(
        self,
        steps: float,
        done: bool = False
    ) -> float:
        """Compute total reward for a day.

        Accumulates step data for clinical outcome and periodically adds
        clinical reward to the daily return. Clinical reward is stored
        and returned only on the day it is computed; otherwise 0.

        Args:
            steps: Daily step count.
            done: Whether episode terminates (will flush clinical reward if any).

        Returns:
            Reward for this day (includes step reward + clinical if due).
        """
        self.day_count += 1
        daily_reward = self.compute_daily_step_reward(steps)

        # Feed steps into clinical simulator
        self.clinical_sim.add_daily_steps(steps)

        clinical_rew = 0.0
        if self.day_count % self.clinical_frequency == 0 or done:
            if len(self.clinical_sim) >= 21 or done:
                clinical_rew = compute_clinical_reward(
                    self.clinical_sim,
                    bp_scale=self.bp_scale,
                    hba1c_scale=self.hba1c_scale
                )
                self.episode_clinical_reward += clinical_rew
                # Reset simulator for next cycle? Or keep sliding window?
                # Keep sliding window: we maintain last 21 days.
                # To avoid double counting, we don't reset; we keep accumulating.
                # This means clinical reward is computed based on rolling 21-day average.
                # That's acceptable.

        return daily_reward + clinical_rew
