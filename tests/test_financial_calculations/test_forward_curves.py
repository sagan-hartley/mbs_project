import unittest
from datetime import datetime
import numpy as np
from financial_calculations.forward_curves import bootstrap_forward_curve


class TestBootstrapForwardCurve(unittest.TestCase):
    """
    Unit test class for testing the `bootstrap_forward_curve` function.

    The tests include:
    - Basic functionality with typical bond inputs.
    - Effect of different initial guesses for the optimization.
    - Handling of edge cases like high coupon rates, long maturities.
    - Consistency of results across multiple runs.
    - Checking equivalency of results when using `datetime` and `datetime64[D]`.
    """

    def setUp(self):
        """
        Set up common data used across multiple tests.
        
        This includes a set of bond data (`cmt_data`), a market close date, 
        a par value for the bonds, and an initial guess for the optimization.
        """
        # Test data for bond maturities and coupon rates
        self.cmt_data = [
            (1, 0.03),  # 1-year bond, 3% coupon rate
            (2, 0.04),  # 2-year bond, 4% coupon rate
            (3, 0.05)   # 3-year bond, 5% coupon rate
        ]
        # Market close date in datetime format
        self.market_close_date = datetime(2024, 8, 10)
        # Par value for the bonds
        self.par_value = 1000
        # Initial guess for the spot rate optimization
        self.initial_guess = 0.03

    def test_basic_bootstrap(self):
        """
        Test basic functionality of `bootstrap_forward_curve` with standard input.
        
        This checks:
        - The length of spot rate dates and rates returned.
        - That spot rates are bounded within the range [0, 1].
        """
        spot_rate_dates, spot_rates = bootstrap_forward_curve(
            self.cmt_data, self.market_close_date, self.par_value
        )

        # Check that the number of spot rate dates equals the number of maturities + 1 (for the market close date)
        self.assertEqual(
            len(spot_rate_dates), len(self.cmt_data) + 1
        )
        # Check that the number of spot rates equals the number of maturities
        self.assertEqual(len(spot_rates), len(self.cmt_data))

        # Spot rates should be between 0 and 1
        self.assertTrue(np.all(np.array(spot_rates) >= 0))
        self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_initial_guess(self):
        """
        Test the effect of varying initial guesses on the results.
        
        This checks whether different initial guesses for the optimization 
        yield consistent outputs within valid bounds.
        """
        # Different initial guesses for the optimization
        initial_guesses = [0.01, 0.05, 0.0, 1.0]
        for guess in initial_guesses:
            spot_rate_dates, spot_rates = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value, initial_guess=guess
            )

            # Check the length of spot rate dates and rates
            self.assertEqual(
                len(spot_rate_dates), len(self.cmt_data) + 1
            )
            self.assertEqual(len(spot_rates), len(self.cmt_data))

            # Spot rates should still be bounded between 0 and 1
            self.assertTrue(np.all(np.array(spot_rates) >= 0))
            self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_edge_cases(self):
        """
        Test edge cases with high coupon rates, low coupon rates, and long maturities.
        
        This ensures the function behaves as expected for extreme values.
        """
        edge_cases = [
            [(1, 0.99)],  # Very high coupon rate
            [(2, 0.1)],   # Low coupon rate
            [(10, 0.01)]  # Long maturity period (10 years)
        ]
        for cmt_data in edge_cases:
            spot_rate_dates, spot_rates = bootstrap_forward_curve(
                cmt_data, self.market_close_date, self.par_value
            )

            # Check the length of spot rate dates and rates
            self.assertEqual(len(spot_rate_dates), len(cmt_data) + 1)
            self.assertEqual(len(spot_rates), len(cmt_data))

            # Spot rates should be bounded between 0 and 1
            self.assertTrue(np.all(np.array(spot_rates) >= 0))
            self.assertTrue(np.all(np.array(spot_rates) <= 1))

    def test_input_types(self):
        """
        Test equivalency of results when the market close date is a `datetime` or `datetime64[D]`.
        
        This ensures the function behaves consistently regardless of the date type.
        """
        # Market close date as datetime and datetime64[D]
        market_close_datetime = self.market_close_date
        market_close_datetime64 = np.datetime64('2024-08-10', 'D')

        # Call the function with both date formats
        spot_dates_datetime, spot_rates_datetime = bootstrap_forward_curve(
            self.cmt_data, market_close_datetime, self.par_value
        )
        spot_dates_datetime64, spot_rates_datetime64 = bootstrap_forward_curve(
            self.cmt_data, market_close_datetime64, self.par_value
        )

        # Ensure outputs are almost equal
        np.testing.assert_array_equal(spot_dates_datetime, spot_dates_datetime64)
        np.testing.assert_array_almost_equal(spot_rates_datetime, spot_rates_datetime64)

    def test_consistency(self):
        """
        Test that the function produces consistent results when run multiple times.
        
        This verifies that the function is deterministic and returns the same results 
        across repeated runs with the same input.
        """
        for _ in range(10):
            spot_rate_dates1, spot_rates1 = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value
            )
            spot_rate_dates2, spot_rates2 = bootstrap_forward_curve(
                self.cmt_data, self.market_close_date, self.par_value
            )

            # Ensure the spot rate dates are exactly the same across runs
            np.testing.assert_array_equal(spot_rate_dates1, spot_rate_dates2)
            # Ensure the spot rates are nearly equal across runs (accounting for floating point precision)
            np.testing.assert_array_almost_equal(spot_rates1, spot_rates2)


if __name__ == '__main__':
    unittest.main()
