"""Unit tests for reward function and clinical outcome."""
import unittest
import numpy as np
from reward import RewardCalculator
from clinical_outcome import simulate_clinical_outcome

class TestClinicalOutcome(unittest.TestCase):
    def test_empty_history(self):
        self.assertEqual(simulate_clinical_outcome([]), 0.0)

    def test_high_steps(self):
        history = [12000] * 21
        score = simulate_clinical_outcome(history)
        self.assertAlmostEqual(score, 1.0)

    def test_low_steps(self):
        history = [2000] * 21
        score = simulate_clinical_outcome(history)
        self.assertAlmostEqual(score, 0.2)

    def test_partial_window(self):
        history = [8000] * 10
        score = simulate_clinical_outcome(history)
        self.assertAlmostEqual(score, 0.8)

class TestRewardCalculator(unittest.TestCase):
    def test_daily_step_reward(self):
        calc = RewardCalculator()
        self.assertEqual(calc.daily_step_reward(5000), 0.5)
        self.assertEqual(calc.daily_step_reward(15000), 1.0)

    def test_step_without_clinical_bonus(self):
        calc = RewardCalculator(clinical_bonus_weight=0.5)
        # First 20 days: no bonus yet
        total = 0
        for _ in range(20):
            reward = calc.step(8000)
            total += reward
        expected_steps = 20 * min(1, 8000/10000) * 1.0  # 16.0
        self.assertAlmostEqual(total, expected_steps)

    def test_step_with_clinical_bonus(self):
        calc = RewardCalculator(clinical_bonus_weight=0.5)
        # 21 days: should trigger bonus
        for _ in range(21):
            calc.step(10000)  # steps_reward = 1.0 each day, steps_since=21
        # After 21 steps, bonus applied. Let's compute manually:
        # steps_history = [10000]*21 => clinical_score = 1.0
        # bonus = 0.5*1.0 = 0.5 added on day 21 reward
        # So total after 21 days: 20 days * 1.0 + day21: 1.0+0.5 = 1.5 => total = 21.5
        self.assertEqual(calc.steps_since_clinical, 0)
        # To verify, we need to check the last reward. We'll recalc.
        # Simpler: create new calc and step 21 times, check cumulative.
        calc2 = RewardCalculator(0.5)
        total = 0
        days = 21
        for _ in range(days):
            total += calc2.step(10000)
        expected = 20 * 1.0 + (1.0 + 0.5)
        self.assertAlmostEqual(total, expected)

    def test_reset(self):
        calc = RewardCalculator()
        calc.step(10000)
        calc.reset()
        self.assertEqual(len(calc.steps_history), 0)
        self.assertEqual(calc.steps_since_clinical, 0)

if __name__ == '__main__':
    unittest.main()
