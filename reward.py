"""Reward function combining daily steps and clinical outcome bonus."""
from clinical_outcome import simulate_clinical_outcome

DAILY_STEPS_WEIGHT = 1.0
CLINICAL_BONUS_WEIGHT = 0.5
CLINICAL_EVAL_PERIOD = 21  # days

class RewardCalculator:
    def __init__(self, clinical_bonus_weight=CLINICAL_BONUS_WEIGHT):
        self.steps_history = []
        self.clinical_bonus_weight = clinical_bonus_weight
        self.steps_since_clinical = 0

    def daily_step_reward(self, steps: int) -> float:
        """Reward based on daily steps (normalized to 0-1)."""
        return min(1.0, steps / 10000.0)

    def step(self, steps: int) -> float:
        """Compute reward for a single day, updating history."""
        self.steps_history.append(steps)
        self.steps_since_clinical += 1
        daily_reward = self.daily_step_reward(steps)
        reward = DAILY_STEPS_WEIGHT * daily_reward
        # Check if clinical evaluation period has passed
        if self.steps_since_clinical >= CLINICAL_EVAL_PERIOD:
            clinical_score = simulate_clinical_outcome(self.steps_history)
            reward += self.clinical_bonus_weight * clinical_score
            self.steps_since_clinical = 0
        return reward

    def reset(self):
        self.steps_history = []
        self.steps_since_clinical = 0
