"""Tests for the dosage variable (Paper Section 5.2).

The dosage variable tracks cumulative treatment exposure:
    X_{t+1} = lambda * X_t + 1{treatment or anti-sedentary suggestion}

where lambda = 0.95 (exponential decay). High dosage indicates the user
has received many recent interventions, which correlates with diminishing
treatment effects and increased burden.
"""

import pytest

from paper_reproduction.heartsteps.dosage import DosageTracker


class TestDosageInitialization:
    def test_starts_at_zero(self):
        tracker = DosageTracker(decay=0.95)
        assert tracker.value == 0.0

    def test_custom_initial_value(self):
        tracker = DosageTracker(decay=0.95, initial_value=2.0)
        assert tracker.value == 2.0


class TestDosageUpdate:
    def test_increases_after_treatment(self):
        tracker = DosageTracker(decay=0.95)
        tracker.update(treatment_delivered=True)
        # X_1 = 0.95 * 0 + 1 = 1.0
        assert tracker.value == pytest.approx(1.0)

    def test_decays_when_no_treatment(self):
        tracker = DosageTracker(decay=0.95, initial_value=1.0)
        tracker.update(treatment_delivered=False)
        # X_1 = 0.95 * 1.0 + 0 = 0.95
        assert tracker.value == pytest.approx(0.95)

    def test_anti_sedentary_increases_dosage(self):
        tracker = DosageTracker(decay=0.95)
        tracker.update(treatment_delivered=False, anti_sedentary_delivered=True)
        # Event = treatment OR anti-sedentary = True
        # X_1 = 0.95 * 0 + 1 = 1.0
        assert tracker.value == pytest.approx(1.0)

    def test_neither_treatment_nor_anti_sedentary(self):
        tracker = DosageTracker(decay=0.95, initial_value=2.0)
        tracker.update(treatment_delivered=False, anti_sedentary_delivered=False)
        # X_1 = 0.95 * 2.0 + 0 = 1.9
        assert tracker.value == pytest.approx(1.9)

    def test_both_treatment_and_anti_sedentary(self):
        """Event is union: treatment OR anti-sedentary. Both = same as one."""
        tracker = DosageTracker(decay=0.95)
        tracker.update(treatment_delivered=True, anti_sedentary_delivered=True)
        # Event = True (union), so X_1 = 0.95 * 0 + 1 = 1.0
        assert tracker.value == pytest.approx(1.0)


class TestDosageAccumulation:
    def test_multiple_treatments_accumulate(self):
        tracker = DosageTracker(decay=0.95)
        for _ in range(5):
            tracker.update(treatment_delivered=True)
        # After 5 treatments: X = sum of 0.95^k for k=0..4 = geometric series
        # = (1 - 0.95^5) / (1 - 0.95) = (1 - 0.77378) / 0.05 = 4.5244
        expected = sum(0.95**k for k in range(5))
        assert tracker.value == pytest.approx(expected, rel=1e-6)

    def test_always_non_negative(self):
        tracker = DosageTracker(decay=0.95, initial_value=0.0)
        for _ in range(20):
            tracker.update(treatment_delivered=False)
        assert tracker.value >= 0.0

    def test_converges_to_low_value_without_treatment(self):
        tracker = DosageTracker(decay=0.95, initial_value=5.0)
        for _ in range(100):
            tracker.update(treatment_delivered=False)
        # After 100 no-treatment steps: 5.0 * 0.95^100 ≈ 5.0 * 0.0059 ≈ 0.03
        assert tracker.value < 0.1


class TestDosageDecayParameter:
    def test_higher_decay_means_slower_decay(self):
        fast = DosageTracker(decay=0.95, initial_value=1.0)
        slow = DosageTracker(decay=0.99, initial_value=1.0)
        fast.update(treatment_delivered=False)
        slow.update(treatment_delivered=False)
        # 0.99 * 1.0 > 0.95 * 1.0
        assert slow.value > fast.value
