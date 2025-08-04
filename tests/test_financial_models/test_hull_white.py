import unittest
import numpy as np
import pandas as pd
from utils import (
    years_from_reference,
    create_regular_dates_grid
)
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

        # Set up a more advanced forward curve to test actual values of the Hull-White simulation
        adv_dates = create_regular_dates_grid("2024-10-01", "2039-10-01")
        adv_rates = np.array([
                0.037152689, 0.037128935, 0.037023894, 0.036950150, 0.036817723, 0.036694537,
                0.036541153, 0.036379749, 0.036206621, 0.035993993, 0.035821335, 0.035561345,
                0.035327796, 0.035046872, 0.034800912, 0.034519662, 0.034301415, 0.034039094,
                0.033837231, 0.033616164, 0.033441544, 0.033261279, 0.033157687, 0.033033966,
                0.032966727, 0.032867582, 0.032810329, 0.032709723, 0.032712051, 0.032678288,
                0.032727890, 0.032802810, 0.032882302, 0.033002311, 0.033121135, 0.033248283,
                0.033349087, 0.033481500, 0.033548198, 0.033644680, 0.033781438, 0.033828332,
                0.033988769, 0.034028321, 0.034113045, 0.034196439, 0.034279111, 0.034418190,
                0.034547958, 0.034691128, 0.034806511, 0.034901733, 0.035025973, 0.035121987,
                0.035277551, 0.035448268, 0.035594763, 0.035795894, 0.035951161, 0.036123720,
                0.036305551, 0.036484735, 0.036674024, 0.036889970, 0.037103384, 0.037297479,
                0.037495734, 0.037618304, 0.037758110, 0.037871465, 0.037921970, 0.038184057,
                0.038356549, 0.038503437, 0.038620151, 0.038680809, 0.038777976, 0.038810834,
                0.038922275, 0.038990273, 0.039054130, 0.039116377, 0.039133121, 0.039170768,
                0.039198293, 0.039257014, 0.039328614, 0.039418949, 0.039505111, 0.039616051,
                0.039672769, 0.039791109, 0.039855200, 0.039957880, 0.040105254, 0.040204305,
                0.040368062, 0.040507569, 0.040613730, 0.040767241, 0.040916601, 0.041048484,
                0.041258544, 0.041402153, 0.041559566, 0.041747338, 0.041897894, 0.042101405,
                0.042346425, 0.042540885, 0.042794073, 0.042999333, 0.043173543, 0.043377961,
                0.043518503, 0.043687666, 0.043832287, 0.043967978, 0.044100426, 0.044234340,
                0.044355315, 0.044483477, 0.044612551, 0.044731461, 0.044877540, 0.045009377,
                0.045139615, 0.045267296, 0.045386141, 0.045491997, 0.045642418, 0.045756685,
                0.045902366, 0.046034770, 0.046123281, 0.046218149, 0.046302105, 0.046370548,
                0.046476574, 0.046569591, 0.046645881, 0.046733122, 0.046782861, 0.046820931,
                0.046881562, 0.046912064, 0.046960170, 0.047014943, 0.047021509, 0.047065301,
                0.047046585, 0.047051823, 0.047028825, 0.047009286, 0.046986697, 0.046960333,
                0.046939068, 0.046912937, 0.046891320, 0.046868599, 0.046843076, 0.046822097,
                0.046794752, 0.046772979, 0.046748643, 0.046727087, 0.046706961, 0.046683387,
                0.046663736, 0.046636769, 0.046612991, 0.046588339, 0.046561760, 0.046542331,
                0.046518816, 0.046500795, 0.046480874, 0.046460978, 0.046441521, 0.046417292,
                0.046417292
            ])
        
        self.advanced_forward_curve = StepDiscounter(dates=adv_dates, rates=adv_rates)
        
        # Parameters for Hull-White model
        self.alpha = 1  # Mean reversion rate
        self.sigma = 0.01  # Volatility of short rate
        self.start_rate = 0.02 # Initial rate
        self.sim_dates = pd.to_datetime(["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"])

    def test_calculate_theta_no_mean_reversion(self):
        """Test calculate_theta when alpha is 0 (no mean reversion)."""
        theta_dates, theta_vals = calculate_theta(self.forward_curve, 0, self.sigma, self.sim_dates)
        
        # Check that theta_vals is close to the derivative of forward rates
        expected_dfdt = np.diff([0.02, 0.02, 0.025, 0.025, 0.03]) / \
            np.diff(years_from_reference(self.forward_curve.market_close_date, self.sim_dates)) # Approximate df/dt
        np.testing.assert_almost_equal(theta_vals[:-1], expected_dfdt, decimal=5)

    def test_calculate_theta_with_mean_reversion(self):
        """Test calculate_theta with non-zero alpha (mean reversion)."""
        theta_dates, theta_vals = calculate_theta(self.forward_curve, self.alpha, self.sigma, self.sim_dates)
        
        # Assert the length of theta_dates matches the length of simulation dates
        self.assertEqual(len(theta_dates), len(self.sim_dates))

        # Assert the length of theta_vals matches the length of theta_dates
        self.assertEqual(len(theta_vals), len(theta_dates))
        
        # Ensure theta values are reasonable (within expected range of test data)
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

    def test_hull_white_simulate_no_antithetic(self):
        """Test simulation without antithetic variates."""
        theta = calculate_theta(self.forward_curve, self.alpha, self.sigma, self.sim_dates)
        sim_dates, r_all, r_avg, r_var = hull_white_simulate(self.alpha, self.sigma, theta, self.start_rate, iterations=10000, antithetic=False)
        
        # Assert the shapes of outputs
        self.assertEqual(r_all.shape, (10000, len(self.sim_dates)))
        self.assertEqual(len(r_avg), len(self.sim_dates))
        self.assertEqual(len(r_var), len(self.sim_dates))

        # Check variance meets the theoretical variance of sigma^2 / (2*alpha) * (1 - e^(-2*alpha*t))
        self.assertAlmostEqual(r_var[-1], self.sigma**2 / (2*self.alpha) * 
                    (1 - np.exp(-2*self.alpha*years_from_reference(self.sim_dates[0], self.sim_dates[-1]))),
                    places=4)

    def test_hull_white_simulate_from_curve(self):
        """Test hull_white_simulate_from_curve end-to-end."""
        dates, r_all, r_avg, r_var = hull_white_simulate_from_curve(self.alpha, self.sigma, self.forward_curve, self.sim_dates, iterations=100)

        # Check that the simulated dates match input dates
        np.testing.assert_array_equal(dates, self.sim_dates)

        # Check output shapes
        self.assertEqual(r_all.shape, (100, len(self.sim_dates)))
        self.assertEqual(len(r_avg), len(self.sim_dates))
        self.assertEqual(len(r_var), len(self.sim_dates))

        # Check that initial rates are close to the start rate
        self.assertTrue(np.allclose(r_all[:, 0], self.forward_curve.rates[0], atol=1e-5))

    def test_hull_white_simulate_from_advanced_curve(self):
        """Test hull_white_simulate_from_curve using the more advanced curve data"""
        dates, r_all, r_avg, r_var = hull_white_simulate_from_curve(self.alpha, self.sigma, self.advanced_forward_curve, self.advanced_forward_curve.dates, iterations=1000)

        np.testing.assert_array_equal(dates, self.advanced_forward_curve.dates)
        np.testing.assert_array_almost_equal(r_avg, self.advanced_forward_curve.rates, decimal=4)
        
        # Check variance meets the theoretical variance of sigma^2 / (2*alpha) * (1 - e^(-2*alpha*t))
        self.assertAlmostEqual(r_var[-1], self.sigma**2 / (2*self.alpha) * 
                    (1 - np.exp(-2*self.alpha*years_from_reference(self.advanced_forward_curve.dates[0], self.advanced_forward_curve.dates[-1]))),
                    places=4)

    def test_hull_white_simulate_odd_iterations_antithetic(self):
        """Test that hull_white_simulate raises an error with odd iterations and antithetic=True."""
        with self.assertRaises(ValueError):
            hull_white_simulate(
                self.alpha,
                self.sigma,
                (self.sim_dates, np.ones(len(self.sim_dates)) * 0.02),
                self.start_rate,
                iterations=101,  # Odd number of iterations
                antithetic=True
            )

if __name__ == "__main__":
    unittest.main()
