import unittest
from datetime import datetime
import numpy as np
from financial_calculations.forward_curves import bootstrap_forward_curve


class TestBootstrapForwardCurve(unittest.TestCase):
    """
    Unit test class for testing the `bootstrap_forward_curve` function.

    Tests include:
    - Basic functionality with typical inputs.
    - Effect of different initial guesses.
    - Edge cases with varying coupon rates and maturities.
    - Consistency of results across multiple runs.
    """

    def setUp(self):
        """
        Set up common data for multiple tests.
        """
        self.cmt_data = [
            (1, 0.03),  # 1-year bond, 3% coupon
            (2, 0.04),  # 2-year bond, 4% coupon
            (3, 0.05)   # 3-year bond, 5% coupon
        ]
        self.market_close_date = datetime(2024, 8, 10)
        self.par_value = 1000
        self.initial_guess = 0.03

    def test_basic_bootstrap(self):
        """
        Test that the function returns a correct structure with valid inputs.
        """
        spot_rate_dates, spot_rates = bootstrap_forward_curve(
            self.cmt_data, self.market_close_date, self.par_value
        )

        # Check that output lengths match the input
        self.assertEqual(
            len(spot_rate_dates), len(self.cmt_data) + 1
        )  # The spot rate dates grid should be 1 index longer because it contains the market close date
        self.assertEqual(len(spot_rates), len(self.cmt_data))

        # Verify that spot rates are bounded by [0, 1]
        self.assertTrue(np.all(np.array(spot_rates) >= 0))
        self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_initial_guess(self):
        """
        Test the effect of varying initial guesses on the bootstrap result.
        """
        initial_guesses = [0.01, 0.05, 0.0, 1.0]
        for guess in initial_guesses:
            spot_rate_dates, spot_rates = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value, initial_guess=guess
            )

            # Check that output lengths match the input
            self.assertEqual(
                len(spot_rate_dates), len(self.cmt_data) + 1
            )  # The spot rate dates grid should be 1 index longer because it contains the market close date
            self.assertEqual(len(spot_rates), len(self.cmt_data))

            # Verify that spot rates are bounded by [0, 1]
            self.assertTrue(np.all(np.array(spot_rates) >= 0))
            self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_edge_cases(self):
        """
        Test edge cases with varying coupon rates and maturities.
        """
        edge_cases = [
            [(1, 0.99)],  # High coupon rate
            [(2, 0.1)],   # Low coupon rate
            [(10, 0.01)]  # Long maturity
        ]
        for cmt_data in edge_cases:
            spot_rate_dates, spot_rates = bootstrap_forward_curve(
                cmt_data, self.market_close_date, self.par_value
            )
            self.assertEqual(len(spot_rate_dates), len(cmt_data) + 1)
            self.assertEqual(len(spot_rates), len(cmt_data))
            self.assertTrue(np.all(np.array(spot_rates) >= 0))
            self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_consistency(self):
        """
        Test that the function produces consistent results across multiple runs.
        """
        for _ in range(10):
            spot_rate_dates1, spot_rates1 = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value
            )
            spot_rate_dates2, spot_rates2 = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value
            )
            np.testing.assert_array_equal(spot_rate_dates1, spot_rate_dates2)
            np.testing.assert_array_almost_equal(spot_rates1, spot_rates2)


if __name__ == '__main__':
    unittest.main()
