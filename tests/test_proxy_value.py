"""Tests for the proxy delayed effect (Paper Section 5.4.2).

The proxy H(x, a) captures the expected future value after taking action a
in dosage state x. The simplified MDP has dosage transitions determined by
the decay rate lambda and the action.

Reference:
    Liao et al. (2019). arXiv:1909.03539, Section 5.4.2.
"""

import numpy as np

from paper_reproduction.heartsteps.proxy_value import ProxyValueFunction


class TestProxyValueProperties:
    def test_eta_positive_all_dosages(self):
        """eta(x) > 0 at all dosage levels (proxy penalises sending)."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            treat_benefit=2.0,
            burden_coef=0.5,
        )
        proxy.solve()
        for x in [0.0, 1.0, 2.0, 5.0, 10.0]:
            assert proxy.eta(x) >= 0.0, f"eta({x}) = {proxy.eta(x)} < 0"

    def test_H_treatment_never_exceeds_no_treatment(self):
        """H(x, 1) <= H(x, 0): treatment never improves future value."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            treat_benefit=2.0,
            burden_coef=0.5,
        )
        proxy.solve()
        for x in [0.0, 1.0, 3.0, 5.0]:
            assert proxy.H(x, 1) <= proxy.H(x, 0), f"H({x},1) > H({x},0)"

    def test_bellman_convergence(self):
        """Value iteration converges: Bellman residual is small."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            treat_benefit=2.0,
            burden_coef=0.5,
        )
        info = proxy.solve(tol=1e-6)
        assert info["converged"]
        assert info["residual"] < 1e-5

    def test_H_values_consistent(self):
        """H(x, a) values are finite and consistent after solve."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            treat_benefit=2.0,
            burden_coef=0.5,
        )
        proxy.solve()
        x_mid = 5.0
        h0 = proxy.H(x_mid, 0)
        h1 = proxy.H(x_mid, 1)
        assert np.isfinite(h0)
        assert np.isfinite(h1)


class TestWeightedUpdate:
    def test_w_zero_reverts_to_H1(self):
        """With w=0, update sets H = H_1 (initial proxy)."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            w=0.0,
        )
        # Set H_1 to known values
        n_grid = len(proxy.grid)
        proxy._H1 = np.ones((n_grid, 2)) * 5.0
        proxy.solve()
        # After update with w=0: H = (1-0)*H1 + 0*H_current = H1
        H1_saved = proxy._H1.copy()
        proxy.update()
        np.testing.assert_allclose(proxy._H_current, H1_saved)

    def test_w_one_blends_fully_to_H_star(self):
        """With w=1, update sets H = H_star from solved Bellman equation."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            w=1.0,
        )
        # First solve to compute H_star
        proxy.solve()
        # The internal _H_current after solve is the raw H_star
        H_star = proxy._H_current.copy()

        # Set H_1 explicitly to be different from H_star
        proxy._H1 = np.zeros_like(H_star)

        # With w=1, update should give H_star (since (1-1)*H1 + 1*H_star = H_star)
        proxy.update()
        np.testing.assert_allclose(proxy._H_current, H_star, atol=1e-10)

    def test_w_half_blends_correctly(self):
        """With w=0.5, update gives midpoint of H1 and H_star."""
        proxy = ProxyValueFunction(
            decay=0.95,
            gamma=0.9,
            p_avail=0.8,
            p_sed=0.2,
            w=0.5,
        )
        # Set H_1 to known values
        n_grid = len(proxy.grid)
        proxy._H1 = np.ones((n_grid, 2)) * 10.0

        proxy.solve()
        H_star = proxy._H_current.copy()

        # Expected: (1-0.5) * 10 + 0.5 * H_star
        expected = 0.5 * 10.0 + 0.5 * H_star
        proxy.update()
        np.testing.assert_allclose(proxy._H_current, expected, atol=1e-10)
