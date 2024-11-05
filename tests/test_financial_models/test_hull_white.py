import unittest
import numpy as np
import pandas as pd
from utils import years_from_reference
from financial_calculations.cash_flows import StepDiscounter
from financial_models.hull_white import (
    calculate_theta,
    hull_white_simulate,
    hull_white_simulate_from_curve
)

class TestHullWhiteModel(unittest.TestCase):
    """Unit tests for the Hull-White model functions."""
    
    def setUp(self):
        """Set up data for Hull-White model tests."""
        # Set up a basic forward curve for testing
        dates = pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"])
        rates = np.array([0.02, 0.025, 0.03])  # Forward rates
        self.forward_curve = StepDiscounter(dates=dates, rates=rates)
        
        # Parameters for Hull-White model
        self.alpha = 0.03  # Mean reversion rate
        self.sigma = 0.015  # Volatility of short rate
        self.start_rate = 0.025  # Initial short rate
        self.sim_dates = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"])

    def test_calculate_theta_no_mean_reversion(self):
        """Test calculate_theta when alpha is 0 (no mean reversion)."""
        theta_dates, theta_vals = calculate_theta(self.forward_curve, 0, self.sigma, self.sim_dates)
        
        # Check that theta_vals is close to the derivative of forward rates
        expected_dfdt = np.gradient([0.02, 0.02, 0.025, 0.025, 0.03],
                                years_from_reference(self.forward_curve.market_close_date, self.sim_dates))  # Approximate df/dt
        np.testing.assert_almost_equal(theta_vals, expected_dfdt, decimal=5)

    def test_calculate_theta_with_mean_reversion(self):
        """Test calculate_theta with non-zero alpha (mean reversion)."""
        theta_dates, theta_vals = calculate_theta(self.forward_curve, self.alpha, self.sigma, self.sim_dates)
        
        # Assert the length of theta_vals matches the length of simulation dates
        self.assertEqual(len(theta_vals), len(self.sim_dates))
        
        # Ensure theta values are reasonable (within expected range)
        self.assertTrue(np.all(theta_vals < 0.1) and np.all(theta_vals > -0.1))

    def test_hull_white_simulate(self):
        """Test Hull-White short rate simulation."""
        sim_dates, r_all, r_avg, r_var = hull_white_simulate(self.alpha, self.sigma, (self.sim_dates, np.ones(len(self.sim_dates)) * 0.02), self.start_rate, iterations=100)

        # Check that output shapes are correct
        self.assertEqual(r_all.shape, (100, len(self.sim_dates)))
        self.assertEqual(len(r_avg), len(self.sim_dates))
        self.assertEqual(len(r_var), len(self.sim_dates))

        # Check initial rates are correctly set to start_rate
        self.assertAlmostEqual(r_all[0, 0], self.start_rate, places=5)

    def test_hull_white_simulate_mo_antithetic(self):
        """Test simulation without antithetic variates."""
        sim_dates, r_all, r_avg, r_var = hull_white_simulate(self.alpha, self.sigma, (self.sim_dates, np.ones(len(self.sim_dates)) * 0.02), self.start_rate, iterations=100, antithetic=False)
        
        # Assert the shapes of outputs
        self.assertEqual(r_all.shape, (100, len(self.sim_dates)))
        self.assertEqual(len(r_avg), len(self.sim_dates))
        self.assertEqual(len(r_var), len(self.sim_dates))

    def test_hull_white_simulate_from_curve(self):
        """Test hull_white_simulate_from_curve end-to-end."""
        dates, r_all, r_avg, r_var = hull_white_simulate_from_curve(self.alpha, self.sigma, self.forward_curve, self.sim_dates, self.start_rate, iterations=100)

        # Check that the simulated dates match input dates
        np.testing.assert_array_equal(dates, self.sim_dates)

        # Check output shapes
        self.assertEqual(r_all.shape, (100, len(self.sim_dates)))
        self.assertEqual(len(r_avg), len(self.sim_dates))
        self.assertEqual(len(r_var), len(self.sim_dates))

        # Check that initial rates are close to the start rate
        self.assertTrue(np.allclose(r_all[:, 0], self.start_rate, atol=1e-5))

if __name__ == "__main__":
    unittest.main()
