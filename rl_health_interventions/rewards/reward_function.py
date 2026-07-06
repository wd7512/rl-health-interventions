from typing import List, Optional
from .clinical_outcome import ClinicalOutcomeGenerator

class RewardFunction:
    """Reward function that combines daily step reward with clinical outcome bonus."""

    def __init__(
        self,
        step_reward_scale: float = 1.0,
        clinical_bonus_scale: float = 10.0,
        clinical_episode_days: int = 21,
    ):
        self.step_reward_scale = step_reward_scale
        self.clinical_bonus_scale = clinical_bonus_scale
        self.clinical_episode_days = clinical_episode_days
        self.clinical_gen = ClinicalOutcomeGenerator()
        self.daily_steps_history: List[float] = []

    def reset(self):
        self.daily_steps_history = []

    def step_reward(self, steps: float) -> float:
        """Reward for daily steps: linear up to 10000 steps, then constant."""
        target = 10000.0
        reward = min(steps / target, 1.0) * self.step_reward_scale
        return reward

    def clinical_bonus(self, daily_steps: List[float]) -> float:
        """Bonus based on clinical outcome from the last 21 days."""
        if len(daily_steps) < self.clinical_episode_days:
            return 0.0
        score = self.clinical_gen.compute_clinical_score(daily_steps)
        return score * self.clinical_bonus_scale

    def update_history(self, steps: float):
        self.daily_steps_history.append(steps)

    def compute_total_reward(self, steps: float, is_episode_end: bool = False) -> float:
        """
        Compute total reward for a day.
        If it's the end of an episode (after clinical_episode_days), include clinical bonus.
        """
        self.update_history(steps)
        daily_reward = self.step_reward(steps)
        total_reward = daily_reward
        if is_episode_end and len(self.daily_steps_history) >= self.clinical_episode_days:
            total_reward += self.clinical_bonus(self.daily_steps_history)
        return total_reward
