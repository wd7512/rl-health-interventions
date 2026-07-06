import unittest
import numpy as np
from rl_health_interventions.rewards.clinical_outcome import ClinicalOutcomeGenerator
from rl_health_interventions.rewards.reward_function import RewardFunction

class TestClinicalOutcome(unittest.TestCase):
    def setUp(self):
        self.gen = ClinicalOutcomeGenerator()

    def test_hba1c_change_low_steps(self):
        steps = [2000] * 21
        change = self.gen.compute_hba1c_change(steps)
        self.assertAlmostEqual(change, 0.2, places=4)  # worsening

    def test_hba1c_change_high_steps(self):
        steps = [10000] * 21
        change = self.gen.compute_hba1c_change(steps)
        self.assertGreater(change, -1.0)  # improvement

    def test_sbp_change_low_steps(self):
        steps = [2000] * 21
        change = self.gen.compute_sbp_change(steps)
        self.assertAlmostEqual(change, 5.0, places=4)

    def test_sbp_change_high_steps(self):
        steps = [12000] * 21
        change = self.gen.compute_sbp_change(steps)
        self.assertAlmostEqual(change, -10.0, places=4)  # capped

    def test_clinical_score_range(self):
        steps = [5000] * 21
        score = self.gen.compute_clinical_score(steps)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.rf = RewardFunction(step_reward_scale=1.0, clinical_bonus_scale=10.0)

    def test_step_reward_max(self):
        reward = self.rf.step_reward(10000)
        self.assertAlmostEqual(reward, 1.0)

    def test_step_reward_half(self):
        reward = self.rf.step_reward(5000)
        self.assertAlmostEqual(reward, 0.5)

    def test_clinical_bonus_not_enough_data(self):
        bonus = self.rf.clinical_bonus([5000]*10)
        self.assertEqual(bonus, 0.0)

    def test_clinical_bonus_enough_data(self):
        steps = [10000]*21
        bonus = self.rf.clinical_bonus(steps)
        self.assertGreater(bonus, 0.0)

    def test_total_reward_end_of_episode(self):
        for _ in range(20):
            self.rf.update_history(8000)
        total = self.rf.compute_total_reward(8000, is_episode_end=True)
        self.assertAlmostEqual(total, self.rf.step_reward(8000) + self.rf.clinical_bonus([8000]*21))

    def test_total_reward_not_end(self):
        for _ in range(20):
            self.rf.update_history(8000)
        total = self.rf.compute_total_reward(8000, is_episode_end=False)
        self.assertAlmostEqual(total, self.rf.step_reward(8000))

if __name__ == '__main__':
    unittest.main()
